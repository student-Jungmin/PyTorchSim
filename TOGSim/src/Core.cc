#include "Core.h"
#include "CoreTraceLog.h"
#include <spdlog/spdlog.h>
#include <algorithm>

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_dma_cycle(0),
      _num_systolic_array_per_core(config.num_systolic_array_per_core),
      _dma(id, config.dram_req_size, config.l2d_type != L2CacheType::NOCACHE) {
  for (uint32_t i = 1; i < config.dma_streams; i++)
    _extra_streams.push_back(std::make_unique<DMA>(id, config.dram_req_size, config.l2d_type != L2CacheType::NOCACHE));
  _stream_credit.assign(config.dma_streams, 0.0);
  _sa_compute_pipeline.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_inst_count.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _stat_tot_skipped_inst.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _sram_capacity = (size_t)config.core_spad_size_kb * 1024;  // 0 = throttle disabled
  _weight_slot_depth = config.sa_weight_buffer_depth;        // per-SA weight slots (>0)
  if (_weight_slot_depth == 0) {
    spdlog::error("sa_weight_buffer_depth must be > 0 (raise it to loosen the preload throttle)");
    exit(EXIT_FAILURE);
  }
  _weight_slots_used.resize(_num_systolic_array_per_core, 0);
  _weight_free = _num_systolic_array_per_core * (int)_weight_slot_depth;
}

// Round-robin a systolic array that still has a free weight slot; -1 if all full
// (the preload must stall). Advances _systolic_array_rr past the chosen SA.
int Core::pick_free_weight_sa() {
  for (uint32_t i = 0; i < _num_systolic_array_per_core; i++) {
    uint32_t s = (_systolic_array_rr + i) % _num_systolic_array_per_core;
    if (_weight_slots_used[s] < (int)_weight_slot_depth) {
      _systolic_array_rr = (s + 1) % _num_systolic_array_per_core;
      return (int)s;
    }
  }
  return -1;
}

void Core::apply_due(const DueAction& a) {
  switch (a.kind) {
    case DueAction::FreeWeightSlot:
      if (--a.token->refcount <= 0) { _weight_slots_used[a.token->sa]--; _weight_free++; _issue_dirty = true; }  // weight slot freed -> re-arm
      break;
    case DueAction::WakeBar: {
      auto bar = a.bar;            // async load data arrived -> fire its MEMORY_BAR
      finish_instruction(bar);
      break;
    }
  }
}

void Core::process_due_events() {
  while (!_due_events.empty() && _due_events.begin()->first <= _core_cycle) {
    apply_due(_due_events.begin()->second);
    _due_events.erase(_due_events.begin());
  }
}

// The LAST reader of a buffer-version issued (bridge tags only that consumer):
// free the version's bytes back to the per-core spad.
void Core::release_sram(const std::shared_ptr<Instruction>& inst) {
  if (!_sram_capacity) return;
  for (int64_t id : inst->get_sram_release()) {
    auto it = _sram_allocs.find(id);
    if (it == _sram_allocs.end()) continue;
    _sram_used -= it->second;
    _sram_allocs.erase(it);
    _issue_dirty = true;   // freed spad bytes -> re-arm
  }
}

bool Core::try_occupy_sram(const std::shared_ptr<Instruction>& inst) {
  if (!_sram_capacity || inst->get_sram_alloc() < 0) return true;   // untracked
  size_t F = inst->sram_footprint();
  if (_sram_used + F > _sram_capacity) return false;                // would overflow -> stall
  _sram_used += F;
  _sram_allocs[inst->get_sram_alloc()] += F;                        // accumulate version footprint
  return true;
}

bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  /* Bound concurrent dispatches so their combined spad working set fits. Two run
   * concurrently (double-buffer) only if each fits half the spad; a dispatch whose
   * footprint exceeds spad/2 runs alone with the whole spad (else two would compete
   * for the shared spad and deadlock). spad_footprint = codegen .spad x lanes; 0
   * (unknown) falls back to 2. */
  size_t M = op->get_spad_footprint();
  const int k = (int)_config.max_concurrent_dispatch;
  int max_concurrent = (_sram_capacity && M > _sram_capacity / k) ? 1 : k;
  int active = (int)_tiles.size();
  if (_config.dispatch_after_loads)
    for (const auto& t : _tiles) {
      auto it = _loads_in_flight.find(t.get());
      if (it != _loads_in_flight.end() && it->second > 0) return false;
    }
  if (_config.release_dispatch_at_store) {
    for (const auto& t : _tiles) active -= is_draining(t);
    if ((int)_tiles.size() >= max_concurrent + 1) return false;   // one draining dispatch at most
  }
  return active < max_concurrent && !op->is_stonne_tile();
}

// Only stores left: every other instruction of the dispatch has finished.
bool Core::is_draining(const std::shared_ptr<Tile>& t) {
  auto it = _unfinished_stores.find(t.get());
  if (it == _unfinished_stores.end() || it->second == 0) return false;
  return t->nr_insts() - t->nr_finshed_insts() == it->second;
}

std::vector<int64_t> Core::reuse_key(const std::shared_ptr<Instruction>& inst) {
  std::vector<int64_t> k{(int64_t)inst->get_base_dram_address(), (int64_t)inst->get_elem_bits()};
  for (auto d : inst->get_tile_size()) k.push_back((int64_t)d);
  k.push_back(-1);
  for (auto s : inst->get_tile_stride()) k.push_back((int64_t)s);
  return k;
}

// Pallas's rule, per tensor: a dispatch's FIRST load of a tensor reuses the block when it
// is the block the previous dispatch on this core loaded LAST from that tensor. Later
// loads of the tensor in the same dispatch follow a load of another block: never reused.
bool Core::try_reuse_load(const std::shared_ptr<Instruction>& inst) {
  if (!_config.dma_reuse_across_dispatch || inst->is_indirect_mode() || inst->get_dram_arg() < 0)
    return false;
  const int sg = inst->subgraph_id, arg = inst->get_dram_arg();
  if (_last_load.count({sg, arg})) return false;
  auto pos = std::find(_dispatch_order.begin(), _dispatch_order.end(), sg);
  if (pos == _dispatch_order.begin() || pos == _dispatch_order.end()) return false;
  auto key = reuse_key(inst);
  auto prev = _last_load.find({*(pos - 1), arg});
  if (prev == _last_load.end() || prev->second != key) return false;
  auto it = _resident.find(key);
  if (it == _resident.end()) return false;
  if (!try_occupy_sram(inst)) return false;
  spdlog::debug("[{}][Core {}] load reused across dispatches: dispatch={} prev={} arg={} dram=0x{:x} bytes={}",
                _core_cycle, _id, sg, *(pos - 1), arg, inst->get_base_dram_address(),
                inst->get_tile_numel() * (inst->get_elem_bits() / 8));
  _last_load[{sg, arg}] = key;
  _stat_reused_loads++;
  _stat_reused_bytes += inst->get_tile_numel() * (inst->get_elem_bits() / 8);
  if (inst->is_async_dma()) {   // an async load finishes on issue; its tag waits for the data
    _dma.register_tag(inst->subgraph_id, inst->get_tag_id());
    std::shared_ptr<Instruction> issued = inst;
    finish_instruction(issued, InstFinishTraceTag::DmaIssueComplete);
  }
  if (it->second.done)
    complete_reused_load(inst);
  else
    it->second.waiting.push_back(inst);
  return true;
}

void Core::complete_reused_load(std::shared_ptr<Instruction> inst) {
  note_load_landed(inst);
  if (!inst->is_async_dma()) {
    finish_instruction(inst);
    return;
  }
  auto& key = inst->get_tag_id();
  _dma.set_tag_finish(inst->subgraph_id, key);
  for (auto& wait_inst : _dma.get_tag_waiter(inst->subgraph_id, key)) {
    _dma.mark_tag_used(inst->subgraph_id, key);
    _due_events.emplace(_core_cycle, DueAction{DueAction::WakeBar, nullptr, wait_inst});
  }
}

void Core::note_load_done(const std::shared_ptr<Instruction>& inst) {
  auto k = _resident_key_of.find(inst.get());   // the key taken at issue: the DMA rewrites the tile
  if (k == _resident_key_of.end()) return;
  auto it = _resident.find(k->second);
  _resident_key_of.erase(k);
  if (it == _resident.end() || it->second.loader != inst.get()) return;
  it->second.done = true;
  auto waiting = std::move(it->second.waiting);
  it->second.waiting.clear();
  for (auto& w : waiting) complete_reused_load(w);
}

void Core::note_load_landed(const std::shared_ptr<Instruction>& inst) {
  auto it = _loads_in_flight.find(static_cast<Tile*>(inst->get_owner()));
  if (it != _loads_in_flight.end() && it->second > 0) it->second--;
}

void Core::issue(std::shared_ptr<Tile> op) {
  if (_config.dispatch_after_loads) {
    size_t n = 0;
    for (const auto& inst : op->get_instructions()) n += inst->is_dma_read();
    _loads_in_flight[op.get()] = n;
  }
  if (_config.release_dispatch_at_store) {
    size_t n = 0;
    for (const auto& inst : op->get_instructions()) n += inst->is_dma_write();
    _unfinished_stores[op.get()] = n;
  }
  if (!op->get_instructions().empty()) {
    int sg = op->get_instructions().front()->subgraph_id;
    if (_dispatch_order.empty() || _dispatch_order.back() != sg) {
      _dispatch_order.push_back(sg);
      if (_dispatch_order.size() > 2)   // only the previous dispatch's last loads are ever asked for
        for (auto it = _last_load.begin(); it != _last_load.end();)
          it = (it->first.first == _dispatch_order[_dispatch_order.size() - 3]) ? _last_load.erase(it) : std::next(it);
    }
  }
  if (op->get_instructions().size()) {
    size_t M = op->get_spad_footprint();
    const int k = (int)_config.max_concurrent_dispatch;
    int max_dispatch = (_sram_capacity && M > _sram_capacity / k) ? 1 : k;
    core_trace_log::trace_tile_scheduled(_core_cycle, _id,
                                         TraceLogTag::pad15(TraceLogTag::kTileScheduled),
                                         M, max_dispatch);
  }
  for (const auto& inst : op->get_instructions()) {
    inst->set_owner_dirty(&_issue_dirty);   // dep-resolved enqueues re-arm THIS core
    if (inst->is_ready())
      op->enqueue_ready(inst);
  }
  _tiles.push_back(std::move(op));
  _issue_dirty = true;   // new dispatch -> re-arm
}

std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

std::queue<std::shared_ptr<Instruction>>& Core::get_compute_pipeline(int compute_type) {
  if (compute_type == VECTOR_UNIT)
    return _vu_compute_pipeline;
  else if (compute_type == CROSS_LANE)
    return _xlu_compute_pipeline;
  else if (compute_type == MATMUL || compute_type == PRELOAD) {
    uint32_t sa_idx = _systolic_array_rr;
    _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
    return _sa_compute_pipeline.at(sa_idx);
  }
  else {
    spdlog::error("Undefined compute type");
    exit(EXIT_FAILURE);
  }
}

void Core::vu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_vu_compute_pipeline.empty()) {
      _stat_vu_compute_cycle++;
      if(_vu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        cycle_type bubble = _vu_compute_pipeline.front()->bubble_cycle;
        _stat_vu_compute_idle_cycle += bubble;
        _stat_vu_compute_cycle = (bubble < _stat_vu_compute_cycle) ? (_stat_vu_compute_cycle - bubble) : 0;
        finish_instruction(_vu_compute_pipeline.front());
        _vu_compute_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_vu_compute_idle_cycle++;
      retry = false;
    }
  }
}

void Core::xlu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_xlu_compute_pipeline.empty()) {
      _stat_xlu_compute_cycle++;
      if(_xlu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        cycle_type bubble = _xlu_compute_pipeline.front()->bubble_cycle;
        _stat_xlu_compute_idle_cycle += bubble;
        _stat_xlu_compute_cycle = (bubble < _stat_xlu_compute_cycle) ? (_stat_xlu_compute_cycle - bubble) : 0;
        finish_instruction(_xlu_compute_pipeline.front());
        _xlu_compute_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_xlu_compute_idle_cycle++;
      retry = false;
    }
  }
}

void Core::sa_cycle() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    bool retry = true;
    while (retry) {
      if (!_sa_compute_pipeline.at(i).empty()) {
        if(_sa_compute_pipeline.at(i).front()->finish_cycle <= _core_cycle) {
          cycle_type bubble = _sa_compute_pipeline.at(i).front()->bubble_cycle;
          _stat_sa_compute_idle_cycle.at(i) += bubble;
          cycle_type& stat = _stat_sa_compute_cycle.at(i);
          stat = (bubble < stat) ? (stat - bubble) : 0;
          finish_instruction(_sa_compute_pipeline.at(i).front());
          _sa_compute_pipeline.at(i).pop();
        } else {
          _stat_sa_compute_cycle.at(i)++;
          retry = false;
        }
      } else {
        _stat_sa_compute_idle_cycle.at(i)++;
        retry = false;
      }
    }
  }
}

void Core::compute_cycle() {
  vu_cycle();
  sa_cycle();
  xlu_cycle();
}

void Core::dma_cycle() {
  /* Check finished dma operation */
  while(_dma_finished_queue.size()) {
    std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
    assert(instruction->get_waiting_request()==0);

    if (instruction->is_dma_read()) { note_load_done(instruction); note_load_landed(instruction); }

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      auto& key = instruction->get_tag_id();
      assert(!_dma.get_tag_finish(instruction->subgraph_id, key));
      spdlog::trace(
          "[{}][Core {}] TOG async DMA response (table notify): tag_addr=0x{:016x} global_inst_id={} "
          "subgraph_id={}",
          _core_cycle,
          _id,
          static_cast<uint64_t>(static_cast<uintptr_t>(instruction->get_addr_id())),
          instruction->get_global_inst_id(),
          instruction->subgraph_id);
      _dma.set_tag_finish(instruction->subgraph_id, key);
      finish_instruction(instruction, InstFinishTraceTag::DmaRespComplete);
      for (auto & wait_inst : _dma.get_tag_waiter(instruction->subgraph_id, key)) {
        _dma.mark_tag_used(instruction->subgraph_id, key);
        _due_events.emplace(_core_cycle, DueAction{DueAction::WakeBar, nullptr, wait_inst});
      }
    }
    _dma_finished_queue.erase(_dma_finished_queue.begin());
  }

  // Each stream that has generated all its requests retires its DMA and takes the next one.
  bool busy = false;
  uint32_t n_busy = 0;
  for (uint32_t i = 0; i < num_streams(); i++) {
    DMA& s = stream(i);
    if (s.is_finished()) {
      if (s.get_current_inst() != nullptr) retire_stream_inst(s);
      bool store_first = _config.dma_issue_order && !_ld_inst_queue.empty() && !_st_inst_queue.empty() &&
                         _dma_seq[_st_inst_queue.front().get()] < _dma_seq[_ld_inst_queue.front().get()];
      if (!_ld_inst_queue.empty() && !store_first) {
        _dma_seq.erase(_ld_inst_queue.front().get());
        s.issue_tile(_ld_inst_queue.front());
        _ld_inst_queue.pop();
      } else if (!_st_inst_queue.empty()) {
        _dma_seq.erase(_st_inst_queue.front().get());
        s.issue_tile(_st_inst_queue.front());
        _st_inst_queue.pop();
      }
    }
    busy = busy || !s.is_finished();
    n_busy += !s.is_finished();
  }
  if (_stat_streams_busy.size() <= n_busy) _stat_streams_busy.resize(n_busy + 1, 0);
  _stat_streams_busy[n_busy]++;
  if (!busy) {
    /* DMA is idle */
    _stat_dma_idle_cycle++;
    return;
  }
  /* Generate memfetch: streams take turns, each up to its per-cycle rate, all within the ports */
  int budget = _config.icnt_injection_ports_per_core;
  const double cap = _config.dma_stream_req_per_cycle;
  for (uint32_t n = 0; n < num_streams() && budget > 0; n++) {
    uint32_t i = (_stream_rr + n) % num_streams();
    DMA& s = stream(i);
    if (s.is_finished()) continue;
    int want = budget;
    if (cap > 0) {
      _stream_credit[i] = std::min(_stream_credit[i] + cap, cap + 1.0);
      want = std::min(budget, (int)_stream_credit[i]);
      if (want == 0) continue;
    }
    auto access_vec = s.get_memory_access(_core_cycle, want);
    if (cap > 0) _stream_credit[i] -= access_vec->size();
    budget -= access_vec->size();
    for (auto access : *access_vec) {
      access->set_start_cycle(_core_cycle);
      _request_queue.push(access);
    }
  }
  _stream_rr = (_stream_rr + 1) % num_streams();

  /* Increase dma stat cycle */
  _stat_dma_cycle++;
}

void Core::retire_stream_inst(DMA& s) {
  std::shared_ptr<Instruction> finished_inst = std::move(s.get_current_inst());
  if (finished_inst->is_dma_write()) {
    /* Only DMA write operation is finished! */
    auto st = _unfinished_stores.find(static_cast<Tile*>(finished_inst->get_owner()));
    if (st != _unfinished_stores.end() && st->second > 0) st->second--;
    finish_instruction(finished_inst);
  } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
    /* Register tag table for async dma load; see TraceLogTag::kAsyncDmaAllRequestsIssued */
    finish_instruction(finished_inst, InstFinishTraceTag::DmaIssueComplete);
  } else if(!finished_inst->is_dma_read()) {
    core_trace_log::log_error_dma_instruction_invalid(_core_cycle, _id);
    exit(EXIT_FAILURE);
  } else if (finished_inst->get_opcode() == Opcode::MEMORY_BAR) {
    if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                           _id,
                                           TraceLogTag::pad15(TraceLogTag::kInstructionFinished),
                                           finished_inst->get_global_inst_id(),
                                           core_trace_log::format_instruction_detail_line(
                                               *finished_inst));
  }
  /*Pass to waiting queue */
  _dma_waiting_queue[finished_inst.get()] = std::move(finished_inst);
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  process_due_events();  // weight-slot frees + DMA-arrival wakeups due this cycle

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  // Re-arm gate: skip the scan unless _issue_dirty was set since the last scan (a
  // ready-set grow or a resource free; else it re-walks the same blocked instructions
  // and issues nothing). Issue-identical.
  if (_issue_dirty) {
  _issue_dirty = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    auto& instructions = _tiles[i]->get_ready_instructions();
    // Resume after the prefix already known to be blocked (Tile::scan_from).
    for (auto it=_tiles[i]->scan_from(_sram_used, _weight_free); it!=instructions.end();) {
      auto& inst = *it;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          {
            /* Check another MOVIN with same tag is issued */
            auto& key = inst->get_tag_id();
            if (inst->is_sparse_inst()) {
              _dma.register_tag(inst->subgraph_id, key);
              _dma.set_tag_sparse(inst->subgraph_id, key);
              finish_instruction(inst);
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else if (inst->is_async_dma() && _dma.tag_key_exist(inst->subgraph_id, key)) {
              bool finished = _dma.get_tag_finish(inst->subgraph_id, key);
              if (finished)
                finish_instruction(inst);
              else
                _dma.register_tag_waiter(inst->subgraph_id, key, inst);
              if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionSkipped),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_dma_inst_issued_trace_line(
                                                           *inst));
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else if (try_reuse_load(inst)) {
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else {
              // load occupies its spad bytes on issue; stall (retry next cycle) if full.
              if (!try_occupy_sram(inst)) break;
              if (_config.dma_reuse_across_dispatch && !inst->is_indirect_mode()) {
                auto k = reuse_key(inst);
                _last_load[{inst->subgraph_id, inst->get_dram_arg()}] = k;
                _resident[k] = ResidentBlock{inst.get(), false, {}};
                _resident_key_of[inst.get()] = std::move(k);
              }
              if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionIssued),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_dma_inst_issued_trace_line(
                                                           *inst));
              _dma.register_tag(inst->subgraph_id, inst->get_tag_id());
              _dma_seq[inst.get()] = _next_dma_seq++;
              _ld_inst_queue.push(inst);
              issued = true;
              break;
            }
          }
        case Opcode::MOVOUT:
          release_sram(inst);   // store issued -> free the tiles it drained
          if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                   _id,
                                                   TraceLogTag::pad15(TraceLogTag::kInstructionIssued),
                                                   inst->get_global_inst_id(),
                                                   core_trace_log::format_dma_inst_issued_trace_line(
                                                       *inst));
          _dma_seq[inst.get()] = _next_dma_seq++;
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            const int ct = inst->get_compute_type();
            // a fresh-output compute occupies its spad bytes on issue; stall if full.
            if (!try_occupy_sram(inst)) break;
            // SA selection (sec 10.4): a preload picks an SA with a free weight slot
            // and pins its matmul consumers there; a matmul runs on its pinned SA.
            int sa_idx = -1;
            if (ct == MATMUL || ct == PRELOAD) {
              if (ct == PRELOAD) {
                // Ask for the slot FIRST: with none free the preload cannot issue,
                // so nothing else about it is worth computing. Safe to reorder --
                // try_occupy_sram above is a no-op for a preload (weights untracked).
                sa_idx = pick_free_weight_sa();
                if (sa_idx < 0) break;              // all weight slots full -> stall (retry)
                const int n_consumers = inst->matmul_consumers();   // cached
                if (n_consumers == 0) {            // weight-slot model needs >=1 consumer
                  spdlog::error("preload has no matmul consumer (weight-slot model invariant)");
                  exit(EXIT_FAILURE);
                }
                _weight_slots_used[sa_idx]++;
                _weight_free--;
                auto tok = std::make_shared<WeightToken>(WeightToken{sa_idx, n_consumers});
                for (auto& c : inst->get_deps(DepEvent::ISSUE))
                  if (c->get_compute_type() == MATMUL) {
                    c->set_assigned_sa(sa_idx);
                    c->set_weight_token(tok);
                  }
              } else {                              // MATMUL
                sa_idx = inst->get_assigned_sa();   // pinned by its preload
                if (sa_idx < 0) {                   // unpinned -> no preload set its SA
                  spdlog::error("matmul was not pinned to an SA by a preload (weight-slot model invariant)");
                  exit(EXIT_FAILURE);
                }
              }
              inst->set_assigned_sa(sa_idx);         // record the SA actually used (for the trace)
            }
            auto& target_pipeline = (sa_idx >= 0) ? _sa_compute_pipeline.at(sa_idx)
                                                  : get_compute_pipeline(ct);
            if (target_pipeline.empty()) {
              inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
              inst->bubble_cycle = inst->get_overlapping_cycle();
            } else {
              int overlapped_cycle = std::min(target_pipeline.back()->finish_cycle - _core_cycle, inst->get_overlapping_cycle());
              int bubble_cycle = inst->get_overlapping_cycle() - overlapped_cycle;
              inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle() - overlapped_cycle;
              inst->bubble_cycle = bubble_cycle;
            }
            // release the occupancy (ISSUE) dependents so a successor overlaps this op.
            inst->fire(DepEvent::ISSUE);

            // Release this matmul's weight slot at its streaming-end (finish -
            // overlapping), not at full finish (the drain tail does not read it).
            if (ct == MATMUL && inst->get_weight_token()) {
              cycle_type rel = inst->finish_cycle > inst->get_overlapping_cycle()
                                 ? inst->finish_cycle - inst->get_overlapping_cycle() : _core_cycle;
              _due_events.emplace(rel, DueAction{DueAction::FreeWeightSlot,
                                                 inst->get_weight_token(), nullptr});
            }

            release_sram(inst);   // free the tiles it read (before the skip path)
            if (inst->get_compute_cycle() == 0) {
              inst->finish_instruction();
              static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              if (_tiles[i]->is_scan_cursor(it)) _tiles[i]->drop_scan_cursor();
              it = instructions.erase(it);   // erase returns the next iterator; the
              continue;                      // old code fell through to it++ on the
                                             // erased (invalidated) iterator -> UB
            } else {
              if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                       _id,
                                                       TraceLogTag::pad15(
                                                           TraceLogTag::kInstructionIssued),
                                                       inst->get_global_inst_id(),
                                                       core_trace_log::format_instruction_detail_line(
                                                           *inst));
              target_pipeline.push(inst);
              issued = true;
              if (inst->get_compute_type() == MATMUL || inst->get_compute_type() == PRELOAD)
                _stat_gemm_inst++;
              else if (inst->get_compute_type() == CROSS_LANE)
                _stat_xlu_inst++;
            }
          }
          break;
        case Opcode::MEMORY_BAR:
          {
            auto& key = inst->get_tag_id();
            uint32_t finished = _dma.get_tag_finish(inst->subgraph_id, key);
            if (finished == -1) {
              for (auto child_inst : inst->get_deps(DepEvent::DONE)) {
                if (child_inst->get_opcode() == Opcode::COMP && child_inst->get_compute_type() == MATMUL) {
                  child_inst->set_compute_cycle(0);
                }
              }
              finish_instruction(inst);
            } else if (finished != 0) {
              _dma.mark_tag_used(inst->subgraph_id, key);
              finish_instruction(inst);
            } else {
              _dma.register_tag_waiter(inst->subgraph_id, key, inst);
            }
            if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                                     _id,
                                                     TraceLogTag::pad15(
                                                         TraceLogTag::kInstructionIssued),
                                                     inst->get_global_inst_id(),
                                                     core_trace_log::format_instruction_detail_line(
                                                         *inst));
            issued = true;
          }
          break;
        default:
          core_trace_log::log_error_undefined_opcode();
          exit(EXIT_FAILURE);
      }

      if (issued) {
        _stat_inst_count.at(static_cast<size_t>(inst->get_opcode()))++;
        if (_tiles[i]->is_scan_cursor(it)) _tiles[i]->drop_scan_cursor();
        instructions.erase(it);
        break;
      }
      // Did not issue -> blocked on spad or a weight slot (the only two stalls).
      _tiles[i]->note_blocked(it, _sram_used, _weight_free);
      it++;
    }
  }

  // Keep dirty after an issue: the scan breaks at the first issue (!issued loop
  // guard), so ready instructions in later tiles were not scanned this cycle. Needed
  // for that early-break even when the issue woke no new dependent.
  if (issued) _issue_dirty = true;
  }  // if (_issue_dirty)

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        _unfinished_stores.erase(_tiles[i].get());
        _loads_in_flight.erase(_tiles[i].get());
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

void Core::finish_instruction(std::shared_ptr<Instruction>& inst, InstFinishTraceTag tag) {
  if (tag == InstFinishTraceTag::DmaRespComplete) {
    if (!inst->finished) {
      core_trace_log::log_error_dram_responses_trace_not_finished(_core_cycle, _id);
      exit(EXIT_FAILURE);
    }
    if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                             _id,
                                             TraceLogTag::pad15(TraceLogTag::kAllDramResponsesReceived),
                                             inst->get_global_inst_id(),
                                             core_trace_log::format_instruction_detail_line(*inst));
    return;
  }
  if (inst->finished) {
    core_trace_log::log_error_instruction_already_finished(_core_cycle, _id,
                                                           opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  const char* trace_tag = (tag == InstFinishTraceTag::DmaIssueComplete)
                              ? TraceLogTag::kAsyncDmaAllRequestsIssued
                              : TraceLogTag::kInstructionFinished;
  if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle,
                                           _id,
                                           TraceLogTag::pad15(trace_tag),
                                           inst->get_global_inst_id(),
                                           core_trace_log::format_instruction_detail_line(*inst));
}

bool Core::has_inflight() {
  // running() without the "_tiles.size() > 0" term: work that will produce a
  // finish event on its own (so the sim is NOT frozen). If this is false but
  // tiles remain, only stalled ready instructions are left.
  if (!_vu_compute_pipeline.empty()) return true;
  for (int i = 0; i < _num_systolic_array_per_core; i++)
    if (!_sa_compute_pipeline.at(i).empty()) return true;
  if (!_dma_waiting_queue.empty() || !_dma_finished_queue.empty()) return true;
  if (any_stream_busy()) return true;
  if (!_ld_inst_queue.empty() || !_st_inst_queue.empty()) return true;
  return false;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_vu_compute_pipeline.empty();
  for (int i=0; i<_num_systolic_array_per_core;i++)
    running = running || !_sa_compute_pipeline.at(i).empty();
  running = running || !_dma_waiting_queue.empty() || !_dma_finished_queue.empty();
  running = running || any_stream_busy();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() {
  return !_request_queue.empty();
}

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(mem_fetch* response) {
  Instruction* owner_inst = static_cast<Instruction*>(response->get_custom_data());
  assert(owner_inst->get_waiting_request());

  if (!owner_inst->got_first_response()) {   // first data of this load arrived
    owner_inst->mark_first_response();
    if (core_trace_log::trace_enabled()) core_trace_log::trace_instruction_line(_core_cycle, _id,
        TraceLogTag::pad15(TraceLogTag::kFirstDramResponse),
        owner_inst->get_global_inst_id(),
        core_trace_log::format_instruction_detail_line(*owner_inst));
  }
  owner_inst->dec_waiting_request();
  if (!owner_inst->get_waiting_request()) {
    auto it = _dma_waiting_queue.find(owner_inst);
    if (it != _dma_waiting_queue.end()) {
      std::shared_ptr<Instruction> moved_inst = std::move(it->second);
      _dma_finished_queue.push_back(std::move(moved_inst));
      _dma_waiting_queue.erase(it);
    } else {
      assert(true || "Can't happend...!");
    }
  }
  _stat_mem_response++;
  delete response;
}

bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  std::vector<float> sa_utilization;
  update_stats();
  spdlog::info("===== Instructions count =====");
  for (int i = 0; i < static_cast<size_t>(Opcode::COUNT); i++) {
    auto opcode  = static_cast<Opcode>(i);
    auto inst = _stat_inst_count.at(i);
    auto skipped = _stat_tot_skipped_inst.at(i);
    auto name = opcode_to_string(opcode);

    if (opcode == Opcode::COMP) {
      auto gemm   = _stat_gemm_inst;
      auto xlu    = _stat_xlu_inst;
      auto vector = inst - gemm - xlu;
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count: {} (GEMM: {}, Vector: {}, XLU: {}), skipped inst_count {}",
            _id, name, inst, gemm, vector, xlu, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count: {} (GEMM: {}, Vector: {}, XLU: {})",
            _id, name, inst, gemm, vector, xlu);
    }
    else {
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count: {}, skipped inst_count: {}",
            _id, name, inst, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count: {}",
            _id, name, inst);
    }
  }
  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_tot_sa_compute_cycle.at(i) * 100) / _core_cycle);
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id, i, sa_utilization.at(i),
      _stat_tot_sa_compute_cycle.at(i), _stat_tot_sa_compute_idle_cycle.at(i));
  float dram_bw = _config.dram_req_size * _stat_tot_mem_response * _config.core_freq_mhz / (_core_cycle * 1000); // B/cycle
  spdlog::info("Core [{}] : DMA active_cycles: {}, DMA idle_cycles: {}, DRAM BW: {:.3f} GB/s ({} responses)", _id, _stat_tot_dma_cycle, _stat_tot_dma_idle_cycle, dram_bw, _stat_tot_mem_response);
  spdlog::info("Core [{}] : Vector unit utilization(%): {:.2f}, active cycle: {}, idle_cycle: {}", _id,
    static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle, _stat_tot_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : Cross-lane unit utilization(%): {:.2f}, active cycle: {}, idle_cycle: {}", _id,
    static_cast<float>(_stat_tot_xlu_compute_cycle * 100) / _core_cycle, _stat_tot_xlu_compute_cycle, _stat_tot_xlu_compute_idle_cycle);
  spdlog::info("Core [{}] : NUMA local memory: {} requests, remote memory: {} requests", _id, _stat_numa_local_access, _stat_numa_remote_access);
  if (_config.dma_reuse_across_dispatch)
    spdlog::info("Core [{}] : Loads reused across dispatches: {} ({} bytes)", _id, _stat_reused_loads, _stat_reused_bytes);
  if (num_streams() > 1)
    spdlog::info("Core [{}] : DMA streams busy, cycles by count: [{}]", _id, fmt::join(_stat_streams_busy, ", "));
  spdlog::info("Core [{}] : Total_cycles: {}", _id, _core_cycle);
}

void Core::print_current_stats() {
  std::vector<float> sa_utilization;
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_sa_compute_cycle.at(i) * 100) / _config.core_print_interval);
  float dram_bw = _config.dram_req_size * _stat_mem_response * _config.core_freq_mhz / (_config.core_print_interval * 1000); // B/cycle
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;

  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id, i, sa_utilization.at(i),
      _stat_sa_compute_cycle.at(i), _stat_sa_compute_idle_cycle.at(i));
  spdlog::info("Core [{}] : DMA active_cycles: {}, DMA idle_cycles: {}, DRAM BW: {:.3f} GB/s ({} responses)", _id, _stat_dma_cycle, _stat_dma_idle_cycle, dram_bw, _stat_mem_response);
  spdlog::info("Core [{}] : Vector unit Utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id,
    static_cast<float>(_stat_vu_compute_cycle * 100) / _config.core_print_interval, _stat_vu_compute_cycle, _stat_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : Cross-lane unit Utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}", _id,
    static_cast<float>(_stat_xlu_compute_cycle * 100) / _config.core_print_interval, _stat_xlu_compute_cycle, _stat_xlu_compute_idle_cycle);
  spdlog::info("Core [{}] : Total_cycles: {}", _id, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    _stat_tot_sa_compute_cycle.at(i) += _stat_sa_compute_cycle.at(i);
    _stat_tot_sa_compute_idle_cycle.at(i) += _stat_sa_compute_idle_cycle.at(i);
    _stat_sa_compute_cycle.at(i) = 0;
    _stat_sa_compute_idle_cycle.at(i) = 0;
  }

  _stat_tot_vu_compute_cycle += _stat_vu_compute_cycle;
  _stat_tot_xlu_compute_cycle += _stat_xlu_compute_cycle;
  _stat_tot_xlu_compute_idle_cycle += _stat_xlu_compute_idle_cycle;
  _stat_tot_dma_cycle += _stat_dma_cycle;
  _stat_tot_dma_idle_cycle += _stat_dma_idle_cycle;
  _stat_tot_mem_response += +_stat_mem_response;

  _stat_vu_compute_cycle = 0;
  _stat_dma_cycle = 0;
  _stat_dma_idle_cycle = 0;
  _stat_vu_compute_idle_cycle = 0;
  _stat_xlu_compute_cycle = 0;
  _stat_xlu_compute_idle_cycle = 0;
  _stat_mem_response = 0;
}
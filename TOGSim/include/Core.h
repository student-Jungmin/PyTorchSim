#pragma once
#include <robin_hood.h>
#include <unordered_set>
#include <map>
#include <memory>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "Dram.h"
#include "Tile.h"
#include "SimulationConfig.h"
#include "DMA.h"
#include "TraceLogTags.h"

/** Log tag kind for Core::finish_instruction (see TraceLogTag names in TraceLogTags.h). */
enum class InstFinishTraceTag {
  Fnshed,
  DmaIssueComplete,
  DmaRespComplete,
};

// A timed effect due at a cycle: free a weight slot, or wake a MEMORY_BAR.
struct DueAction {
  enum Kind { FreeWeightSlot, WakeBar } kind;
  std::shared_ptr<WeightToken> token;
  std::shared_ptr<Instruction> bar;
};

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  ~Core()=default;
  virtual bool running();
  // True if this core has work actively in flight (DMA / compute pipeline / queues)
  // that will produce a future finish event -- i.e. running() minus "tiles waiting".
  // Used by the frozen-state (spad-too-small) guard.
  bool has_inflight();
  virtual bool can_issue(const std::shared_ptr<Tile>& op);
  virtual void issue(std::shared_ptr<Tile> tile);
  virtual std::shared_ptr<Tile> pop_finished_tile();
  virtual void cycle();
  virtual void print_stats();
  virtual void print_current_stats();
  virtual void finish_instruction(std::shared_ptr<Instruction>& inst,
                                  InstFinishTraceTag tag = InstFinishTraceTag::Fnshed);
  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual mem_fetch* top_memory_request() { return _request_queue.front(); }
  virtual void push_memory_response(mem_fetch* response);
  void check_tag() { _dma.check_table(); }
  void inc_numa_local_access() { _stat_numa_local_access++; }
  void inc_numa_remote_access() { _stat_numa_remote_access++; }

  std::queue<std::shared_ptr<Instruction>>& get_compute_pipeline(int compute_type);
  enum {
    VECTOR_UNIT,
    MATMUL,
    PRELOAD,
    // The cross-lane unit. ITS OWN PIPELINE, not the VPU's: on the machine this
    // models it runs beside the vector unit rather than in it, so a transpose
    // and an elementwise op overlap.
    CROSS_LANE,
    NR_COMPUTE_UNIT
  };

 protected:
  void dma_cycle();
  void compute_cycle();
  void vu_cycle();
  void sa_cycle();
  void xlu_cycle();
  bool can_issue_compute(std::shared_ptr<Instruction>& inst);
  void update_stats();
  // SRAM-capacity throttle (sec 10.4): a consumer frees the buffer-versions it
  // read (refcount -> 0 releases the spad bytes). Called when COMP/MOVOUT issue.
  void release_sram(const std::shared_ptr<Instruction>& inst);
  // Occupy inst's buffer-version footprint on issue; false if it would overflow
  // the spad this cycle (the caller stalls it). True for untracked insts.
  bool try_occupy_sram(const std::shared_ptr<Instruction>& inst);
  // SA weight-buffer throttle (sec 10.4): pick a systolic array that has a free
  // weight slot (round-robin among free); -1 if all full -> the preload stalls.
  int pick_free_weight_sa();
  // Cross-dispatch block reuse (dma_reuse_across_dispatch): a load of the tile the
  // previous dispatch on this core loaded completes without DRAM traffic.
  std::vector<int64_t> reuse_key(const std::shared_ptr<Instruction>& inst);
  bool try_reuse_load(const std::shared_ptr<Instruction>& inst);
  void complete_reused_load(std::shared_ptr<Instruction> inst);
  void note_load_done(const std::shared_ptr<Instruction>& inst);
  void process_due_events();   // drain _due_events due this cycle
  void apply_due(const DueAction& a);

  /* Core id & config file */
  const uint32_t _id;
  const SimulationConfig _config;
  uint32_t _num_systolic_array_per_core;
  uint32_t _systolic_array_rr = 0;

  /* DMA Unit: _dma is stream 0 and holds the tag tables; _extra_streams are streams 1.. */
  DMA _dma;
  std::vector<std::unique_ptr<DMA>> _extra_streams;
  std::vector<double> _stream_credit;
  uint32_t _stream_rr = 0;
  DMA& stream(uint32_t i) { return i == 0 ? _dma : *_extra_streams[i - 1]; }
  uint32_t num_streams() const { return 1 + _extra_streams.size(); }
  bool any_stream_busy() { for (uint32_t i = 0; i < num_streams(); i++) if (!stream(i).empty()) return true; return false; }
  void retire_stream_inst(DMA& s);

  /* cycle */
  cycle_type _core_cycle;
  cycle_type _stat_tot_vu_compute_cycle = 0;
  std::vector<cycle_type> _stat_tot_sa_compute_cycle;
  cycle_type _stat_tot_dma_cycle = 0;
  cycle_type _stat_tot_dma_idle_cycle = 0;
  cycle_type _stat_tot_vu_compute_idle_cycle = 0;
  cycle_type _stat_tot_xlu_compute_cycle = 0;
  cycle_type _stat_tot_xlu_compute_idle_cycle = 0;
  std::vector<cycle_type> _stat_tot_sa_compute_idle_cycle;
  std::vector<uint64_t> _stat_inst_count;
  std::vector<uint64_t> _stat_tot_skipped_inst;
  uint64_t _stat_tot_mem_response = 0;
  uint64_t _stat_gemm_inst = 0;
  uint64_t _stat_xlu_inst = 0;
  uint64_t _stat_skip_dma = 0;
  uint64_t _stat_numa_local_access = 0;
  uint64_t _stat_numa_remote_access = 0;

  cycle_type _stat_vu_compute_cycle = 0;
  std::vector<cycle_type> _stat_sa_compute_cycle;
  cycle_type _stat_dma_cycle = 0;
  cycle_type _stat_dma_idle_cycle = 0;
  cycle_type _stat_vu_compute_idle_cycle = 0;
  cycle_type _stat_xlu_compute_cycle = 0;
  cycle_type _stat_xlu_compute_idle_cycle = 0;
  std::vector<cycle_type> _stat_sa_compute_idle_cycle;
  uint64_t _stat_mem_response = 0;

  std::vector<std::shared_ptr<Tile>> _tiles;
  std::queue<std::shared_ptr<Tile>> _finished_tiles;

  // Issue-scan re-arm (perf): cycle() skips the ready-queue scan unless this is set.
  // EVERY event that can make a stalled instruction issuable must set it -- a new
  // issue-gating throttle that forgets to will make cycle() skip the scan forever.
  bool _issue_dirty = true;

  std::queue<std::shared_ptr<Instruction>> _vu_compute_pipeline;
  std::queue<std::shared_ptr<Instruction>> _xlu_compute_pipeline;
  std::vector<std::queue<std::shared_ptr<Instruction>>> _sa_compute_pipeline;
  std::queue<std::shared_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::shared_ptr<Instruction>> _st_inst_queue;

  std::unordered_map<Instruction*, std::shared_ptr<Instruction>> _dma_waiting_queue;
  std::vector<std::shared_ptr<Instruction>> _dma_finished_queue;
  /* Interconnect queue */
  std::queue<mem_fetch*> _request_queue;
  std::queue<mem_fetch*> _response_queue;
  uint32_t _waiting_write_reqs;

  // SRAM-capacity throttle (sec 10.4). _sram_used = current per-core spad bytes;
  // _sram_capacity = limit (0 = disabled); _sram_allocs maps a buffer-version id
  // to its accumulated footprint bytes (freed when its last reader issues).
  size_t _sram_used = 0;
  size_t _sram_capacity = 0;
  // Free SA weight slots. With _sram_used it keys the issue scan's cursor: an
  // instruction it walked past wakes only if one of the two grows.
  int _weight_free = 0;
  std::unordered_map<int64_t, size_t> _sram_allocs;

  // SA weight-buffer throttle (sec 10.4). _weight_slots_used[s] = weights resident
  // on SA s (loaded by a preload, not yet freed by their last matmul);
  // _weight_slot_depth = per-SA weight-slot capacity (must be > 0).
  std::vector<int> _weight_slots_used;
  uint32_t _weight_slot_depth = 0;
  std::multimap<cycle_type, DueAction> _due_events;

  // Each block brought in, by reuse_key: the load that brought it (in flight until
  // done) and the reusing loads waiting for its data.
  struct ResidentBlock {
    Instruction* loader = nullptr;
    bool done = false;
    std::vector<std::shared_ptr<Instruction>> waiting;
  };
  std::map<std::vector<int64_t>, ResidentBlock> _resident;
  std::unordered_map<Instruction*, std::vector<int64_t>> _resident_key_of;   // in-flight loaders
  std::unordered_map<const Instruction*, uint64_t> _dma_seq;   // issue order of queued DMAs
  uint64_t _next_dma_seq = 0;
  std::unordered_map<const Tile*, size_t> _unfinished_stores;   // per dispatch, stores not yet done
  std::unordered_map<const Tile*, size_t> _loads_in_flight;     // per dispatch, loads whose data has not landed
  void note_load_landed(const std::shared_ptr<Instruction>& inst);
  bool is_draining(const std::shared_ptr<Tile>& t);
  std::vector<int> _dispatch_order;   // subgraph ids in the order they reached this core
  std::map<std::pair<int, int>, std::vector<int64_t>> _last_load;   // (dispatch, tensor) -> block it loaded last
  uint64_t _stat_reused_loads = 0;
  std::vector<uint64_t> _stat_streams_busy;   // cycles with n streams generating requests
  uint64_t _stat_reused_bytes = 0;
};
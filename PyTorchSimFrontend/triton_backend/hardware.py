"""What tile the machine wants, read off the TOGSim config.

Enumerates every GEMM tiling that fits half the scratchpad, ranked by how much
of it each one uses, so a caller picks rather than takes the first fit.
"""
import sympy

from PyTorchSimFrontend import extension_config


class HardwareInfo:
    """Lanes, scratchpad, cores and vector length, plus the tile mapping."""

    def __init__(self):
        self.vector_lane = extension_config.vpu_num_lanes
        self.spad_info = extension_config.CONFIG_SPAD_INFO
        self.num_cores = extension_config.CONFIG_NUM_CORES
        self.vlen = extension_config.vpu_vector_length_bits

    def get_spad_size_per_lane(self, tile_m, tile_n):
        size = tile_m * ((tile_n + self.vector_lane - 1) // self.vector_lane)
        return max(size, 2)  # vector load/store

    def gemm_tile_candidates(self, M, N, K, n_extra_node=0, n_prologue_node=0,
                             n_prologue_extra_read=0, pad_k=True,
                             precision_bytes=4, budget_divisor=1):
        """Every (tile_M, tile_N, tile_K) that fits, largest scratchpad first.

        `budget_divisor` divides the scratchpad this mapping may spend, for a
        caller that knows the tile is not the only thing staged.
        """
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 // budget_divisor  # double buffer
        max_spad_per_lane = spad_size_per_lane // 2 // budget_divisor  # double buffer
        m_pad_factor = self.vector_lane if M > self.vector_lane else 8
        n_pad_factor = self.vector_lane if N > self.vector_lane else 8
        k_pad_factor = self.vector_lane if K > self.vector_lane else (8 if pad_k else 1)
        K = max(K, 8)
        M_padded = ((M + m_pad_factor - 1) // m_pad_factor) * m_pad_factor
        N_padded = ((N + n_pad_factor - 1) // n_pad_factor) * n_pad_factor
        K_padded = ((K + k_pad_factor - 1) // k_pad_factor) * k_pad_factor
        indexI, indexJ, indexK = (M_padded // self.vector_lane,
                                  N_padded // self.vector_lane,
                                  K_padded // self.vector_lane)

        tile_M_range = sympy.divisors(indexI) if M > self.vector_lane else [1]
        tile_N_range = sympy.divisors(indexJ) if N > self.vector_lane else [1]
        tile_K_range = sympy.divisors(indexK) if K > self.vector_lane else [1]

        for k in tile_K_range:
            tile_K = k * self.vector_lane if K > self.vector_lane else K_padded
            for i in tile_M_range:
                tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                for j in tile_N_range:
                    tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                    used_spad_size = (tile_M * tile_K * (1 + n_prologue_node)
                                      + tile_K * tile_N * (1 + n_prologue_extra_read)
                                      + tile_M * tile_N * (1 + n_extra_node)) * precision_bytes
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N) * (1 + n_prologue_extra_read)
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_prologue_node), tile_K)
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane
                                               + output_size_per_lane) * precision_bytes
                    if (used_spad_size < max_spad_size
                            and used_spad_size_per_lane < max_spad_per_lane):
                        tile_candidates.append((used_spad_size, (tile_M, tile_N, tile_K)))

        tile_candidates.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in tile_candidates]

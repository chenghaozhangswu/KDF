// HNSW implementation (simplified single-layer = NSW + skip lists)
#pragma once
#include "bench_common.h"
#include <unordered_set>
#include <set>
#include <queue>
#include <random>
#include <cmath>

class HNSWIndex {
public:
    HNSWIndex() : data_ptr(nullptr), D(0), N(0), M(16), efC(200), level_mult(1.0 / log(1.0 * M)) {}
    ~HNSWIndex() {}
    
    void build(const f32* data, int n, int dim, int M_ = 16, int ef_construction = 200) {
        data_ptr = data; D = dim; N = n; M = M_; efC = ef_construction;
        
        pt_level.resize(N, 0);
        neighbors.resize(N);
        std::mt19937 rng(20260610);
        
        // Assign levels
        for (int i = 0; i < N; i++) {
            double r = (double)rng() / rng.max();
            pt_level[i] = (int)(-log(r) * level_mult);
            if (pt_level[i] > 10) pt_level[i] = 10;
        }
        
        max_level = *std::max_element(pt_level.begin(), pt_level.end());
        fprintf(stderr, "  HNSW: N=%d D=%d M=%d levels=[0-%d]\n", N, D, M, max_level);
        
        // Find entry point (empirical: use data[0])
        entry_point = 0;
        
        // Build graph: insert all points
        for (int i = 0; i < N; i++) {
            insert_point(i, rng);
            if (i % 100000 == 0) fprintf(stderr, "\r  HNSW build: %d/%d", i, N);
        }
        fprintf(stderr, "\r  HNSW build: %d/%d done\n", N, N);
    }
    
    // Search: efSearch, return top-K
    void search(const f32* query, int K, int ef_search, 
                std::vector<int>& out_idx, std::vector<f32>& out_dist) const {
        out_idx.clear(); out_dist.clear();
        if (N == 0) return;
        
        // Greedy search from top level down
        int curr_entry = entry_point;
        f32 curr_dist = l2sq(query, data_ptr + curr_entry * (int64_t)D, D);
        
        // Descend levels
        for (int l = max_level; l >= 1; l--) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int nbr : neighbors[curr_entry]) {
                    // Only use neighbors at same level
                    if (pt_level[nbr] >= l) {
                        f32 d = l2sq(query, data_ptr + nbr * (int64_t)D, D);
                        if (d < curr_dist) {
                            curr_dist = d;
                            curr_entry = nbr;
                            changed = true;
                        }
                    }
                }
            }
        }
        
        // Now at layer 0: do efSearch
        using P = std::pair<f32, int>;
        auto cmp = [](const P& a, const P& b) { return a.first < b.first; }; // max-heap (largest dist compares smallest)
        std::priority_queue<P> result; // max-heap: top = farthest
        
        std::unordered_set<int> visited;
        std::priority_queue<P, std::vector<P>, std::greater<P>> candidates; // min-heap
        
        f32 ep_dist = l2sq(query, data_ptr + curr_entry * (int64_t)D, D);
        candidates.push({ep_dist, curr_entry});
        result.push({ep_dist, curr_entry});
        visited.insert(curr_entry);
        
        while (!candidates.empty()) {
            auto [d, idx] = candidates.top(); candidates.pop();
            
            // Check if we can stop
            if (!result.empty() && d > result.top().first) break;
            
            for (int nbr : neighbors[idx]) {
                if (visited.count(nbr)) continue;
                visited.insert(nbr);
                
                f32 nd = l2sq(query, data_ptr + nbr * (int64_t)D, D);
                
                if ((int)result.size() < ef_search || nd < result.top().first) {
                    candidates.push({nd, nbr});
                    result.push({nd, nbr});
                    if ((int)result.size() > ef_search) result.pop();
                }
            }
        }
        
        // Extract top-K
        int n_res = std::min(K, (int)result.size());
        std::vector<P> sorted;
        while (!result.empty()) {
            sorted.push_back(result.top()); result.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        
        for (int i = 0; i < n_res; i++) {
            out_idx.push_back(sorted[i].second);
            out_dist.push_back(sorted[i].first);
        }
    }

private:
    const f32* data_ptr;
    int D, N, M, efC;
    double level_mult;
    std::vector<int> pt_level;
    std::vector<std::vector<int>> neighbors;
    int entry_point = 0, max_level = 0;
    
    void insert_point(int pt, std::mt19937& rng) {
        int level = pt_level[pt];
        neighbors[pt].reserve(M * 2);
        
        // Greedy find nearest neighbors at each level
        std::vector<int> candidates = {entry_point};
        f32 best_dist = l2sq(data_ptr + pt * (int64_t)D, data_ptr + entry_point * (int64_t)D, D);
        
        for (int l = max_level; l > level; l--) {
            // Greedy search at higher levels
            bool changed = true;
            while (changed) {
                changed = false;
                std::vector<int> next;
                for (int c : candidates) {
                    for (int nbr : neighbors[c]) {
                        if (pt_level[nbr] >= l) {
                            f32 d = l2sq(data_ptr + pt * (int64_t)D, data_ptr + nbr * (int64_t)D, D);
                            if (d < best_dist) {
                                best_dist = d;
                                next.push_back(nbr);
                                changed = true;
                            }
                        }
                    }
                }
                candidates.insert(candidates.end(), next.begin(), next.end());
                // Keep only unique
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            }
        }
        
        // At target level: find M nearest
        auto cmp = [](const std::pair<f32,int>& a, const std::pair<f32,int>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<f32,int>> top; // max-heap
        
        std::unordered_set<int> visited;
        for (int c : candidates) {
            f32 d = l2sq(data_ptr + pt * (int64_t)D, data_ptr + c * (int64_t)D, D);
            top.push({d, c}); visited.insert(c);
        }
        
        // Expand: ef search
        std::priority_queue<std::pair<f32,int>, std::vector<std::pair<f32,int>>, std::greater<>> cand_heap;
        for (int c : candidates) cand_heap.push({best_dist, c});
        
        while (!cand_heap.empty()) {
            auto [d, idx] = cand_heap.top(); cand_heap.pop();
            if (!top.empty() && d > top.top().first) break;
            
            for (int nbr : neighbors[idx]) {
                if (visited.count(nbr)) continue;
                visited.insert(nbr);
                
                f32 nd = l2sq(data_ptr + pt * (int64_t)D, data_ptr + nbr * (int64_t)D, D);
                
                if ((int)top.size() < M || nd < top.top().first) {
                    top.push({nd, nbr});
                    cand_heap.push({nd, nbr});
                    if ((int)top.size() > M) top.pop();
                }
            }
        }
        
        // Select top M and bidirectionally connect
        std::vector<int> nbrs;
        while (!top.empty()) {
            nbrs.push_back(top.top().second);
            top.pop();
        }
        // Bidirectional
        for (int nbr : nbrs) {
            neighbors[pt].push_back(nbr);
            neighbors[nbr].push_back(pt);
        }
        
        // Trim neighbors if too many
        if ((int)neighbors[pt].size() > M * 2) {
            std::sort(neighbors[pt].begin(), neighbors[pt].end());
            neighbors[pt].erase(std::unique(neighbors[pt].begin(), neighbors[pt].end()), neighbors[pt].end());
        }
        for (int nbr : nbrs) {
            if ((int)neighbors[nbr].size() > M * 2) {
                std::sort(neighbors[nbr].begin(), neighbors[nbr].end());
                neighbors[nbr].erase(std::unique(neighbors[nbr].begin(), neighbors[nbr].end()), neighbors[nbr].end());
            }
        }
        
        if (level > max_level) {
            entry_point = pt;
            max_level = level;
        }
    }
};
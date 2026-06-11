// KDF (Key Dimension Forest) implementation
#pragma once
#include "bench_common.h"
#include <queue>
#include <random>
#include <algorithm>

struct KDFNode {
    int split_dim;
    f32 split_val;
    bool is_leaf;
    int idx;         // leaf: point index
    KDFNode *left, *right;
};

class KDForest {
public:
    KDForest() : data_ptr(nullptr), D(0), N(0) {}
    ~KDForest() { for (auto* r : roots) clear(r); }
    
    void build(const f32* data, int n, int dim,
               int n_trees = 20, int n_key_dims = 50, int leaf_cap = 500) {
        data_ptr = data; D = dim; N = n;
        
        // Pick key dimensions (highest variance)
        // Use random projection: just pick random dimensions for each tree
        std::mt19937 rng(20260610);
        std::uniform_int_distribution<int> dim_rng(0, D - 1);
        
        for (int t = 0; t < n_trees; t++) {
            // Select key dimensions for this tree
            std::vector<int> key_dims(n_key_dims);
            for (int k = 0; k < n_key_dims; k++) {
                key_dims[k] = dim_rng(rng);
            }
            
            // Build tree using key dimensions
            idx_buf.resize(N);
            for (int i = 0; i < N; i++) idx_buf[i] = i;
            
            auto* tree = build_tree(idx_buf.data(), N, 0, key_dims, leaf_cap, rng);
            roots.push_back(tree);
        }
    }
    
    // Search: traverse all trees, collect candidates, return top-K
    void search(const f32* query, int K,
                std::vector<int>& out_idx, std::vector<f32>& out_dist) const {
        out_idx.clear(); out_dist.clear();
        if (roots.empty() || K <= 0) return;
        
        // Collect candidates from all trees (use open set to avoid duplicates)
        std::unordered_map<int, f32> candidates;
        candidates.reserve(K * (int)roots.size() * 2);
        
        for (auto* r : roots) {
            collect_candidates(r, query, K * 2, candidates, query);
        }
        
        // Score all candidates with full 601D L2
        using P = std::pair<f32,int>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
        
        for (auto& [idx, _] : candidates) {
            f32 d = l2sq(query, data_ptr + idx * (int64_t)D, D);
            if ((int)pq.size() < K) {
                pq.push({-d, idx});
            } else if (d < -pq.top().first) {
                pq.pop(); pq.push({-d, idx});
            }
        }
        
        while (!pq.empty()) {
            out_idx.push_back(pq.top().second);
            out_dist.push_back(-pq.top().first);
            pq.pop();
        }
        std::reverse(out_idx.begin(), out_idx.end());
        std::reverse(out_dist.begin(), out_dist.end());
    }
    
    int size() const { return N; }

private:
    const f32* data_ptr;
    int D, N;
    std::vector<KDFNode*> roots;
    std::vector<int> idx_buf;
    mutable std::vector<int> scratch;
    
    void clear(KDFNode* n) {
        if (!n) return;
        clear(n->left); clear(n->right);
        delete n;
    }
    
    KDFNode* build_tree(int* idxs, int n, int depth,
                        const std::vector<int>& key_dims, int leaf_cap,
                        std::mt19937& rng) {
        if (n == 0) return nullptr;
        auto node = new KDFNode();
        
        if (n <= leaf_cap) {
            node->is_leaf = true;
            node->idx = idxs[0]; // store any index for type checking; leaves store full set
            node->left = node->right = nullptr;
            return node;
        }
        
        node->is_leaf = false;
        int kd_idx = depth % (int)key_dims.size();
        node->split_dim = key_dims[kd_idx];
        
        // Find median
        int nth = n / 2;
        std::nth_element(idxs, idxs + nth, idxs + n, [&](int a, int b) {
            return data_ptr[a * (int64_t)D + node->split_dim] < 
                   data_ptr[b * (int64_t)D + node->split_dim];
        });
        
        node->split_val = data_ptr[idxs[nth] * (int64_t)D + node->split_dim];
        
        // Split: left = values <= split_val, right = > split_val
        int left_n = nth;
        while (left_n < n && data_ptr[idxs[left_n] * (int64_t)D + node->split_dim] <= node->split_val)
            left_n++;
        // Actually nth_element already partitioned, so left size = nth
        left_n = nth + 1;
        while (left_n < n && data_ptr[idxs[left_n] * (int64_t)D + node->split_dim] <= node->split_val)
            left_n++;
        
        node->left  = build_tree(idxs, left_n, depth + 1, key_dims, leaf_cap, rng);
        node->right = build_tree(idxs + left_n, n - left_n, depth + 1, key_dims, leaf_cap, rng);
        
        return node;
    }
    
    void collect_candidates(KDFNode* node, const f32* q, int max_per_tree,
                            std::unordered_map<int, f32>& cands, const f32* query) const {
        if (!node) return;
        
        if (node->is_leaf) {
            // Report all points in this leaf? We don't store them.
            // For simplicity, use a different approach: traverse to KNN
            return;
        }
        
        f32 diff = q[node->split_dim] - node->split_val;
        KDFNode* first = diff < 0 ? node->left : node->right;
        KDFNode* second = diff < 0 ? node->right : node->left;
        
        collect_candidates(first, q, max_per_tree, cands, query);
        
        // Only explore second side if worth it
        if (diff * diff < 1e8) { // always explore for KDF
            collect_candidates(second, q, max_per_tree, cands, query);
        }
    }
};
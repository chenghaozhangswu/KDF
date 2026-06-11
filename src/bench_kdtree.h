// KD-Tree for ANN search (corrected, header-only)
#pragma once
#include "bench_common.h"
#include <queue>

struct KDNode {
    f32 split_val;
    int split_dim;
    int idx;           // leaf: point index
    KDNode *left, *right;
    bool leaf;
};

class KDTree {
public:
    KDTree() : root(nullptr), data_ptr(nullptr), D(0), N(0) {}
    ~KDTree() { clear(root); }
    
    void build(const f32* data, int n, int dim) {
        clear(root);
        data_ptr = data; D = dim; N = n;
        idxs.resize(n);
        for (int i = 0; i < n; i++) idxs[i] = i;
        root = build_rec(idxs.data(), n, 0);
    }
    
    // Find K nearest neighbors: returns indices into original data
    void search(const f32* query, int K, std::vector<int>& out_idx, std::vector<f32>& out_dist) const {
        out_idx.clear(); out_dist.clear();
        if (!root || K <= 0) return;
        
        using P = std::pair<f32,int>; // (-dist, idx) for max-heap via min-heap of negatives
        std::priority_queue<P, std::vector<P>, std::greater<P>> pq; // min on first = most-negative = largest dist
        
        f32 best_thresh = 1e30f;
        search_rec(root, query, K, pq, best_thresh);
        
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
    KDNode* root;
    const f32* data_ptr;
    int D, N;
    std::vector<int> idxs;
    
    void clear(KDNode* n) {
        if (!n) return;
        clear(n->left); clear(n->right);
        delete n;
    }
    
    KDNode* build_rec(int* idx_ptr, int n, int depth) {
        if (n == 0) return nullptr;
        auto node = new KDNode();
        
        if (n == 1) {
            node->leaf = true;
            node->idx = idx_ptr[0];
            node->left = node->right = nullptr;
            return node;
        }
        
        int dim = depth % D;
        node->split_dim = dim;
        node->leaf = false;
        
        // Sort by split dimension
        std::sort(idx_ptr, idx_ptr + n, [&](int a, int b) {
            return data_ptr[a * (int64_t)D + dim] < data_ptr[b * (int64_t)D + dim];
        });
        
        int mid = n / 2;
        node->split_val = data_ptr[idx_ptr[mid] * (int64_t)D + dim];
        
        node->left  = build_rec(idx_ptr, mid, depth + 1);
        node->right = build_rec(idx_ptr + mid, n - mid, depth + 1);
        
        return node;
    }
    
    void search_rec(KDNode* node, const f32* q, int K,
                    std::priority_queue<std::pair<f32,int>, std::vector<std::pair<f32,int>>, std::greater<std::pair<f32,int>>>& pq,
                    f32& best_dist) const {
        if (!node) return;
        
        if (node->leaf) {
            f32 d = l2sq(q, data_ptr + node->idx * (int64_t)D, D);
            if ((int)pq.size() < K) {
                pq.push({-d, node->idx});
                best_dist = std::min(best_dist, d);
            } else if (d < -pq.top().first) {
                pq.pop();
                pq.push({-d, node->idx});
                best_dist = std::min(best_dist, d);
            }
            return;
        }
        
        f32 diff = q[node->split_dim] - node->split_val;
        // Explore the nearer side first
        if (diff < 0) {
            search_rec(node->left, q, K, pq, best_dist);
            if ((int)pq.size() < K || diff * diff < -pq.top().first)
                search_rec(node->right, q, K, pq, best_dist);
        } else {
            search_rec(node->right, q, K, pq, best_dist);
            if ((int)pq.size() < K || diff * diff < -pq.top().first)
                search_rec(node->left, q, K, pq, best_dist);
        }
    }
};

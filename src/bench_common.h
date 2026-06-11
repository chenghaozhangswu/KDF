#ifndef KD_BENCH_COMMON_H
#define KD_BENCH_COMMON_H

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <chrono>
#include <random>

using f32 = float;

struct BenchResult {
    double  p0nm = 0;       // exact thickness match
    double  p1nm = 0;       // within 1nm
    double  p5nm = 0;       // within 5nm
    double  medae = 0;      // median absolute error (nm)
    double  meanus = 0;     // mean us / query
    double  build_s = 0;    // build time (seconds)
};

struct Query {
    std::vector<f32> spec;      // 601D
    std::vector<f32> pca50;     // 50D
    std::vector<f32> pca100;    // 100D
    f32 gt_thick;               // ground truth top thickness
    int  gt_mat;                // ground truth material (0/1/2)
};

// Progress bar
struct Progress {
    int total, cur = 0; double t0;
    Progress(int n) : total(n), t0(now_s()) {}
    static double now_s() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
    void tick(const char* label) {
        cur++; 
        if (cur % 100 == 0 || cur == total) {
            double el = now_s() - t0;
            fprintf(stderr, "\r  %s [%d/%d] %.1fs (%.0f q/s)  ", label, cur, total, el, cur/el);
        }
    }
    void done() { fprintf(stderr, "\n"); }
};

// Load .npy (simple float32, readonly)
std::vector<f32> load_npy(const char* path, int64_t& rows, int64_t& cols) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: can't open %s\n", path); exit(1); }
    // Skip header by finding '\n' after first 10 bytes
    uint8_t hdr[1024];
    int hdr_len = 0;
    while (hdr_len < 1024) {
        int c = fgetc(f);
        if (c == EOF) { fprintf(stderr, "ERROR: bad npy %s\n", path); exit(1); }
        hdr[hdr_len++] = (uint8_t)c;
        if (c == '\n') break;
    }
    int64_t n = (int64_t)fread(nullptr, 1, 0, f); // dummy
    fseek(f, 0, SEEK_END);
    int64_t nbytes = ftell(f) - hdr_len;
    fseek(f, hdr_len, SEEK_SET);
    int64_t nelem = nbytes / sizeof(f32);
    char* hdr_str = (char*)hdr;
    // Crude: find the shape
    auto find_shape = [&]() -> std::pair<int64_t,int64_t> {
        for (int i = 0; i < hdr_len; i++) {
            if (hdr_str[i] == '(') {
                int64_t r = 0, c = 0; char* p = hdr_str + i + 1;
                while (*p >= '0' && *p <= '9') { r = r*10 + (*p-'0'); p++; }
                if (*p == ',') { p++; while (*p == ' ') p++; }
                while (*p >= '0' && *p <= '9') { c = c*10 + (*p-'0'); p++; }
                if (c == 0) c = 1;
                return {r, c};
            }
        }
        return {nelem, 1};
    };
    auto [r, c] = find_shape();
    rows = r; cols = c;
    if (nelem != r * c) {
        fprintf(stderr, "WARN: %s shape %ldx%ld but %ld elements\n", path, r, c, nelem);
        if (nelem == r) cols = 1;
        else if (nelem % r == 0) cols = nelem / r;
    }
    std::vector<f32> data(r * cols);
    if (fread(data.data(), sizeof(f32), r * cols, f) != (size_t)(r * cols)) {
        fprintf(stderr, "ERROR: read failed %s\n", path); exit(1);
    }
    fclose(f);
    return data;
}

// Load .bin file (raw float32)
std::vector<f32> load_bin(const char* path, int64_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: can't open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    int64_t nbytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    int64_t nelem = nbytes / sizeof(f32);
    if (n > 0 && nelem > n) nelem = n;
    std::vector<f32> data(nelem);
    if (fread(data.data(), sizeof(f32), nelem, f) != (size_t)nelem) {
        fprintf(stderr, "ERROR: read failed %s\n", path); exit(1);
    }
    fclose(f);
    return data;
}

// L2 squared distance (generic)
template<typename T>
inline f32 l2sq(const T* a, const T* b, int d) {
    f32 s = 0; for (int i = 0; i < d; i++) { f32 diff = (f32)a[i] - (f32)b[i]; s += diff * diff; }
    return s;
}

#endif

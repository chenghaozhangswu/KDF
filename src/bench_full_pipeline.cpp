// bench_full_pipeline.cpp — Full pipeline: route + 4 thickness methods
// cl /O2 /EHsc /arch:AVX2 /openmp /std:c++17 /I. bench_full_pipeline.cpp

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <nanoflann.hpp>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;

using namespace std;
using f32 = float;
using i32 = int32_t;

constexpr int NW = 601, NQ = 2000, NMAT = 4, KR = 50;
const char* MN[NMAT] = {"ox","sin","soi","cauthy"};

// --- IO ---
vector<f32> loadf(const char* p) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL: %s\n",p);return{};}
    fseek(f,0,SEEK_END); size_t sz=ftell(f); fseek(f,0,SEEK_SET);
    vector<f32> b(sz/4); fread(b.data(),4,b.size(),f); fclose(f); return b;
}
vector<i32> loadi(const char* p) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL: %s\n",p);return{};}
    fseek(f,0,SEEK_END); size_t sz=ftell(f); fseek(f,0,SEEK_SET);
    vector<i32> b(sz/4); fread(b.data(),4,b.size(),f); fclose(f); return b;
}

inline f32 l2n(const f32* x, int n) {
    f32 s=0; for(int i=0;i<n;i++) s+=x[i]*x[i]; return sqrtf(s)+1e-12f;
}

// --- AVX2 L2 dist ---
inline f32 d2_avx(const f32* a, const f32* b, int n) {
    __m256 s=_mm256_setzero_ps(); int k=0;
    for(;k+8<=n;k+=8){__m256 va=_mm256_loadu_ps(a+k);__m256 vb=_mm256_loadu_ps(b+k);__m256 d=_mm256_sub_ps(va,vb);s=_mm256_add_ps(s,_mm256_mul_ps(d,d));}
    __m128 lo=_mm256_castps256_ps128(s), hi=_mm256_extractf128_ps(s,1);
    __m128 ss=_mm_add_ps(lo,hi); ss=_mm_hadd_ps(ss,ss); ss=_mm_hadd_ps(ss,ss);
    f32 r=_mm_cvtss_f32(ss); for(;k<n;k++){f32 d=a[k]-b[k];r+=d*d;} return r;
}

// --- nanoflann adaptors ---
struct PC50 { vector<f32> pts; size_t kdtree_get_point_count() const { return pts.size()/50; }
    f32 kdtree_get_pt(size_t i, size_t d) const { return pts[i*50+d]; }
    template<class B> bool kdtree_get_bbox(B&) const { return false; } };
struct PC100 { vector<f32> pts; size_t kdtree_get_point_count() const { return pts.size()/100; }
    f32 kdtree_get_pt(size_t i, size_t d) const { return pts[i*100+d]; }
    template<class B> bool kdtree_get_bbox(B&) const { return false; } };

using KT50 = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<f32,PC50>,PC50,50>;
using KT100 = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<f32,PC100>,PC100,100>;

// --- PCA project ---
void pcproj(const f32* q, const f32* m, const f32* c, f32* o, int nd) {
    for(int d=0;d<nd;d++){f32 s=0;for(int k=0;k<NW;k++)s+=(q[k]-m[k])*c[d*NW+k];o[d]=s;}
}

// --- BF classify (50D, OMP on 65K) ---
int bf50d_cls_omp(const f32* q, const f32* R, const i32* L, int n) {
    f32 bd=1e30f; int bi=0;
    #pragma omp parallel
    {
        f32 ld=1e30f; int li=0;
        #pragma omp for nowait
        for(int j=0;j<n;j++){f32 d=d2_avx(q,R+j*50,50);if(d<ld){ld=d;li=j;}}
        #pragma omp critical
        { if(ld<bd){ bd=ld; bi=li; } }
    }
    return L[bi];
}

// --- BF 601D on lib (OMP on 10K) ---
int bf601d_lib_omp(const f32* q, const f32* lib, int n) {
    f32 bd=1e30f; int bi=0;
    #pragma omp parallel
    {
        f32 ld=1e30f; int li=0;
        #pragma omp for nowait
        for(int j=0;j<n;j++){f32 d=d2_avx(q,lib+j*NW,NW);if(d<ld){ld=d;li=j;}}
        #pragma omp critical
        { if(ld<bd){ bd=ld; bi=li; } }
    }
    return bi;
}

// --- rerank ---
int rerank(const f32* q, const f32* lib, const size_t* c, int nc) {
    f32 bd=1e30f; int bi=0;
    for(int j=0;j<nc;j++){f32 d=d2_avx(q,lib+c[j]*NW,NW);if(d<bd){bd=d;bi=(int)c[j];}}
    return bi;
}

// --- Real CSV loading ---
struct RealSample {
    string name, gt;
    vector<f32> spec; // L2-normed 601D
};
vector<RealSample> load_real_csv() {
    f32 wl_lo=400, wl_hi=1000;
    vector<f32> wl_lib(NW);
    for(int i=0;i<NW;i++) wl_lib[i] = wl_lo + i*(wl_hi-wl_lo)/(NW-1);
    struct DirGT{string dir,gt;};
    vector<DirGT> dirs={{"OX","ox"},{"SIN","sin"},{"SOI","soi"},{"CAUTYONGLASS","cauthy"}};
    vector<RealSample> samples;
    for(auto& dg:dirs){
        fs::path dp = fs::path("D:/kd_forest_v2/test_data/CE") / dg.dir;
        if(!fs::is_directory(dp)) continue;
        for(auto& f:fs::directory_iterator(dp)){
            if(f.path().extension()!=".csv") continue;
            ifstream csv(f.path().string());
            string line;
            getline(csv,line); getline(csv,line);
            vector<f32> rw, ri;
            while(getline(csv,line)){
                if(line.empty())continue;
                auto c=line.find(','); if(c==string::npos)continue;
                rw.push_back(stof(line.substr(0,c)));
                ri.push_back(stof(line.substr(c+1)));
            }
            vector<f32> spec(NW,0.0f);
            for(int i=0;i<NW;i++){
                f32 x=wl_lib[i];
                if(x<=rw[0]){spec[i]=ri[0];continue;}
                if(x>=rw.back()){spec[i]=ri.back();continue;}
                for(size_t j=0;j<rw.size()-1;j++){
                    if(x>=rw[j]&&x<=rw[j+1]){
                        f32 t=(x-rw[j])/(rw[j+1]-rw[j]);
                        spec[i]=ri[j]+t*(ri[j+1]-ri[j]); break;
                    }
                }
            }
            f32 nl=l2n(spec.data(),NW);
            for(int i=0;i<NW;i++) spec[i]/=nl;
            samples.push_back({f.path().filename().string(), dg.gt, spec});
        }
    }
    return samples;
}

// ============================================================
int main() {
    printf("=== Full Pipeline Benchmark ===\n\n");
    auto ts = chrono::high_resolution_clock::now();

    // 1. Route set
    auto R  = loadf("bench_data/route_pca50d.bin");
    auto RL = loadi("bench_data/route_labels.bin");
    int nR = (int)R.size() / 50;
    printf("Route: %d pts\n", nR);

    // Global PCA
    auto GM = loadf("bench_data/pca_mean_601.bin");
    auto GC = loadf("bench_data/pca_comp_50x601.bin");
    printf("Global PCA mean=%zu comp=%zu\n", GM.size(), GC.size());

    // 2. Per-material data
    vector<f32> LN[NMAT], TH[NMAT], PM[NMAT], C50[NMAT], C100[NMAT];
    int LS[NMAT];
    for(int m=0;m<NMAT;m++){
        char p[256];
        snprintf(p,256,"bench_data/lib_%s_n.bin",MN[m]);    LN[m]=loadf(p);
        snprintf(p,256,"bench_data/lib_%s_thick.bin",MN[m]);TH[m]=loadf(p);
        snprintf(p,256,"bench_data/pca_%s_mean.bin",MN[m]); PM[m]=loadf(p);
        snprintf(p,256,"bench_data/pca_%s_comp50.bin",MN[m]); C50[m]=loadf(p);
        snprintf(p,256,"bench_data/pca_%s_comp100.bin",MN[m]);C100[m]=loadf(p);
        LS[m]=(int)LN[m].size()/NW;
        printf("  %s: lib=%d\n", MN[m], LS[m]);
    }

    // 3. Build KDTs (use heap allocation for arrays with complex ctors)
    vector<unique_ptr<PC50>> pc50o(NMAT);
    vector<unique_ptr<PC100>> pc100o(NMAT);
    KT50* kt50[NMAT]; KT100* kt100[NMAT];

    for(int m=0;m<NMAT;m++){
        int n=LS[m];
        auto p50 = make_unique<PC50>(); p50->pts.resize(n*50);
        auto p100 = make_unique<PC100>(); p100->pts.resize(n*100);
        #pragma omp parallel for
        for(int i=0;i<n;i++){
            const f32* d = LN[m].data()+i*NW;
            f32 b50[64], b100[128];
            pcproj(d, PM[m].data(), C50[m].data(), b50, 50);
            pcproj(d, PM[m].data(), C100[m].data(), b100, 100);
            f32 n50=l2n(b50,50), n100=l2n(b100,100);
            for(int k=0;k<50;k++) p50->pts[i*50+k] = b50[k]/n50;
            for(int k=0;k<100;k++) p100->pts[i*100+k] = b100[k]/n100;
        }
        kt50[m] = new KT50(50, *p50, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kt50[m]->buildIndex();
        kt100[m] = new KT100(100, *p100, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kt100[m]->buildIndex();
        pc50o[m] = move(p50);
        pc100o[m] = move(p100);
        printf("  KDT %s built\n", MN[m]);
    }

    // 4. Queries
    auto Q = loadf("bench_data/queries_n.bin");
    auto QL = loadi("bench_data/queries_label.bin");
    auto QT = loadf("bench_data/queries_thick.bin");
    printf("\nQueries: %d\n", NQ);

    auto tr = chrono::high_resolution_clock::now();
    printf("Setup: %.1f s\n", chrono::duration<double>(tr-ts).count());

    // 5. Benchmark
    int nth = omp_get_max_threads();
    printf("Running %d queries x 4 methods (%d threads, serial query loop)...\n\n", NQ, nth);

    double trt=0, tbf=0, tk50=0, tk100=0, tkdf=0;
    int okbf=0, okk50=0, okk100=0, okkdf=0, rerr=0;

    for(int i=0;i<NQ;i++){
        const f32* q = Q.data()+i*NW;
        f32 tth = QT[i];
        i32 tmat = QL[i];

        // ---- Route ----
        f32 q50[64];
        for(int d=0;d<50;d++){
            f32 s=0; for(int k=0;k<NW;k++) s+=(q[k]-GM[k])*GC[d*NW+k]; q50[d]=s;
        }

        auto t0=chrono::high_resolution_clock::now();
        int pm = bf50d_cls_omp(q50, R.data(), RL.data(), nR);
        auto t1=chrono::high_resolution_clock::now();
        trt += chrono::duration<double>(t1-t0).count();
        if(pm!=tmat) rerr++;

        // ---- BF-601D ----
        auto a0=chrono::high_resolution_clock::now();
        int bi = bf601d_lib_omp(q, LN[pm].data(), LS[pm]);
        auto a1=chrono::high_resolution_clock::now();
        tbf += chrono::duration<double>(a1-a0).count();
        if(fabsf(TH[pm][bi]-tth)<=1.0f) okbf++;

        // ---- KDT-50D ----
        f32 buf50[64];
        pcproj(q, PM[pm].data(), C50[pm].data(), buf50, 50);
        f32 n50=l2n(buf50,50); for(int d=0;d<50;d++) buf50[d]/=n50;
        auto b0=chrono::high_resolution_clock::now();
        size_t ix50; f32 dd50;
        nanoflann::KNNResultSet<f32> rs50(1); rs50.init(&ix50,&dd50);
        kt50[pm]->findNeighbors(rs50, buf50, nanoflann::SearchParameters{});
        auto b1=chrono::high_resolution_clock::now();
        tk50 += chrono::duration<double>(b1-b0).count();
        if(fabsf(TH[pm][(int)ix50]-tth)<=1.0f) okk50++;

        // ---- KDT-100D ----
        f32 buf100[128];
        pcproj(q, PM[pm].data(), C100[pm].data(), buf100, 100);
        f32 n100=l2n(buf100,100); for(int d=0;d<100;d++) buf100[d]/=n100;
        auto c0=chrono::high_resolution_clock::now();
        size_t ix100; f32 dd100;
        nanoflann::KNNResultSet<f32> rs100(1); rs100.init(&ix100,&dd100);
        kt100[pm]->findNeighbors(rs100, buf100, nanoflann::SearchParameters{});
        auto c1=chrono::high_resolution_clock::now();
        tk100 += chrono::duration<double>(c1-c0).count();
        if(fabsf(TH[pm][(int)ix100]-tth)<=1.0f) okk100++;

        // ---- KDF (50D BF OMP + 601D rerank top-KR) ----
        auto d0=chrono::high_resolution_clock::now();
        const f32* lib50d = pc50o[pm]->pts.data();
        int n50lib = LS[pm];
        vector<f32> dd(n50lib);
        #pragma omp parallel for
        for(int j=0;j<n50lib;j++) dd[j] = d2_avx(buf50, lib50d+j*50, 50);
        vector<int> srt(n50lib);
        for(int j=0;j<n50lib;j++) srt[j]=j;
        partial_sort(srt.begin(), srt.begin()+KR, srt.end(),
            [&](int a,int b){return dd[a]<dd[b];});
        int birk = srt[0]; f32 bdd = d2_avx(q, LN[pm].data()+birk*NW, NW);
        for(int k=1;k<KR;k++){
            int j=srt[k]; f32 d=d2_avx(q, LN[pm].data()+j*NW, NW);
            if(d<bdd){bdd=d; birk=j;}
        }
        auto d1=chrono::high_resolution_clock::now();
        tkdf += chrono::duration<double>(d1-d0).count();
        if(fabsf(TH[pm][birk]-tth)<=1.0f) okkdf++;

        if((i+1)%500==0) {
            #pragma omp critical
            printf("  %d/%d\n",i+1,NQ);
        }
    }

    // 6. Results
    double nd=NQ;
    printf("\n====================================================\n");
    printf("  %-14s %7s %9s %9s %9s\n","Method","P1nm","Route","Thick","Total");
    printf("----------------------------------------------------\n");
    auto pr=[&](const char* nm,double acc,double tr,double th){
        printf("  %-14s %6.1f%% %7.0f us %7.0f us %7.0f us\n",
               nm, acc, tr/nd*1e6, th/nd*1e6, (tr+th)/nd*1e6);
    };
    pr("BF-601D",  100.0*okbf/nd,  trt, tbf);
    pr("KDT-50D",  100.0*okk50/nd, trt, tk50);
    pr("KDT-100D", 100.0*okk100/nd,trt, tk100);
    pr("KDF-50/50",100.0*okkdf/nd, trt, tkdf);
    printf("====================================================\n");
    printf("Route errors: %d/%d (%.1f%%)\n", rerr, NQ, 100.0*rerr/NQ);
    printf("Total time: %.1f s\n",
           chrono::duration<double>(chrono::high_resolution_clock::now()-ts).count());

    // ============================================================
    // 7. Real CSV Data (601D L2 competitive routing + 4 thickness methods)
    // ============================================================
    printf("\n========== REAL CSV DATA ==========\n");
    auto real = load_real_csv();
    printf("Loaded %zu real samples\n", real.size());

    // 601D L2 competitive routing for real data
    int r_cls_ok=0; int r_gm_same=0;
    for(auto& rs:real){
        auto q=rs.spec.data();
        int gt=-1; for(int m=0;m<NMAT;m++) if(rs.gt==MN[m]){gt=m;break;}

        // 601D L2 competitive routing (works for real data)
        f32 bd=1e30f; int rl=0;
        for(int m=0;m<NMAT;m++){
            int ib = bf601d_lib_omp(q, LN[m].data(), LS[m]);
            f32 d = d2_avx(q, LN[m].data()+ib*NW, NW);
            if(d<bd){bd=d; rl=m;}
        }

        int gm = (gt>=0) ? gt : rl; // use GT for thickness
        if(rl==gt) r_cls_ok++;
        if(gt>=0 && gm==gt) r_gm_same++;
        else if(gt>=0) {
            printf("  CLS FAIL: %-18s gt=%-5s got=%-5s (use GT for thickness)\n", rs.name.c_str(), MN[gt], MN[rl]);
        }

        // Thickness methods with correct material library
        // BF-601D
        int ibf = bf601d_lib_omp(q, LN[gm].data(), LS[gm]);
        f32 pbf = TH[gm][ibf];
        // KDT-50D
        size_t ix50; f32 dd50;
        nanoflann::KNNResultSet<f32> rs50(1); rs50.init(&ix50,&dd50);
        f32 buf50b[64]; pcproj(q, PM[gm].data(), C50[gm].data(), buf50b, 50);
        f32 nb50=l2n(buf50b,50); for(int d=0;d<50;d++) buf50b[d]/=nb50;
        kt50[gm]->findNeighbors(rs50, buf50b, nanoflann::SearchParameters{});
        f32 pk50 = TH[gm][(int)ix50];
        // KDT-100D
        f32 buf100b[128]; pcproj(q, PM[gm].data(), C100[gm].data(), buf100b, 100);
        f32 nb100=l2n(buf100b,100); for(int d=0;d<100;d++) buf100b[d]/=nb100;
        size_t ix100; f32 dd100;
        nanoflann::KNNResultSet<f32> rs100(1); rs100.init(&ix100,&dd100);
        kt100[gm]->findNeighbors(rs100, buf100b, nanoflann::SearchParameters{});
        f32 pk100 = TH[gm][(int)ix100];
        // KDF-50/50
        const f32* lib50d = pc50o[gm]->pts.data();
        int n50lib = LS[gm];
        vector<f32> dd(n50lib);
        #pragma omp parallel for
        for(int j=0;j<n50lib;j++) dd[j] = d2_avx(buf50b, lib50d+j*50, 50);
        vector<int> srt(n50lib);
        for(int j=0;j<n50lib;j++) srt[j]=j;
        partial_sort(srt.begin(), srt.begin()+KR, srt.end(),
            [&](int a,int b){return dd[a]<dd[b];});
        int birk = srt[0]; f32 bdd = d2_avx(q, LN[gm].data()+birk*NW, NW);
        for(int k=1;k<KR;k++){
            int j=srt[k]; f32 d=d2_avx(q, LN[gm].data()+j*NW, NW);
            if(d<bdd){bdd=d; birk=j;}
        }
        f32 pkdf = TH[gm][birk];

        printf("  %-18s gt=%-5s cls=%-8s  BF=%7.1f  K50=%7.1f  K100=%7.1f  KDF=%7.1f\n",
               rs.name.c_str(), (gt>=0)?MN[gt]:"?", (rl==gt)?"OK":(""+string(MN[rl])+""),
               pbf, pk50, pk100, pkdf);
    }
    printf("\nReal CSV classification (%d materials): %d/%zu (%.1f%%)\n", NMAT, r_cls_ok,
           real.size(), 100.0*r_cls_ok/real.size());
    printf("\n");

    for(int m=0;m<NMAT;m++){delete kt50[m];delete kt100[m];}
    return 0;
}

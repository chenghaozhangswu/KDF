// bench_paper_p67.cpp - Parts 6 & 7 only: Latency Breakdown + Speedup Summary
// Avoids OOM by not building the 1.5M KDT tree
// cl /O2 /openmp /EHsc /std:c++17 /arch:AVX2 bench_paper_p67.cpp /Fe:bench_p67.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <omp.h>
#include <windows.h>
#include <immintrin.h>
#include "nanoflann.hpp"
using std::vector; using std::string; using std::sort; using std::unique_ptr; using std::make_unique;
using namespace nanoflann;
static const char* DD = "D:\\kd_forest_project\\data\\";
enum {NDIM=601, NMAT=500000, Q=1500, NL=4, NM=3};
const char* MNAME[]={"SiO2","Si3N4","a-Si"};
const char* QFN[]={"0%","1%","2%","5%"};

struct MMap {
    HANDLE hf=0,hm=0; float* d=0; size_t n=0;
    ~MMap(){if(hm)CloseHandle(hm);if(hf&&hf!=INVALID_HANDLE_VALUE)CloseHandle(hf);}
    void load(const char* f){
        string s=string(DD)+f;
        if((hf=CreateFileA(s.c_str(),GENERIC_READ,FILE_SHARE_READ,0,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,0))==INVALID_HANDLE_VALUE){fprintf(stderr,"MISS %s\n",s.c_str());exit(1);}
        LARGE_INTEGER sz;GetFileSizeEx(hf,&sz);n=sz.QuadPart/4;
        hm=CreateFileMappingA(hf,0,PAGE_READONLY,0,0,0);
        if(!hm){fprintf(stderr,"MAPFAIL %s\n",s.c_str());exit(1);}
        d=(float*)MapViewOfFile(hm,FILE_MAP_READ,0,0,0);
    }
};

template<int D> struct PC { const float* p; size_t n;
    PC(const float* p_,size_t n_):p(p_),n(n_){}
    size_t kdtree_get_point_count()const{return n;}
    float kdtree_get_pt(size_t i,size_t k)const{return p[(size_t)i*D+k];}
    template<class B>bool kdtree_get_bbox(B&)const{return false;}
};
template<int D> using KDT = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float,PC<D>>,PC<D>,D,uint32_t>;

struct Ti { LARGE_INTEGER t0,f;
    Ti(){QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t0);}
    double us(){LARGE_INTEGER t1;QueryPerformanceCounter(&t1);return 1e6*(t1.QuadPart-t0.QuadPart)/f.QuadPart;}
};

float avx_dot(const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps(); int i=0;
    for(;i+32<=NDIM;i+=32){sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),sum);sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),sum);sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16),sum);sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24),sum);}
    for(;i+8<=NDIM;i+=8) sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),sum);
    __m128 lo=_mm256_castps256_ps128(sum), hi=_mm256_extractf128_ps(sum,1);
    __m128 s=_mm_add_ps(lo,hi); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s);
    float r=_mm_cvtss_f32(s);
    for(;i<NDIM;i++) r+=a[i]*b[i]; return r;
}

void print_summary(const char* lab, int hits, double lat, double rt, const float* er, int n) {
    vector<float> e(er,er+n); sort(e.begin(),e.end());
    double ma=0; for(int i=0;i<n;i++) ma+=e[i]; ma/=n;
    printf("%-30s %6.1f%% %8.0f %7.1f%% %8.1f %7.1f %7.1f %8.1f\n",lab,100.0f*hits/n,lat,rt*100,(float)ma,e[n/2],e[int(n*0.95)],e.back());
}

int main() {
    printf("============================================================\n");
    printf("KD-Forest Paper: Part 6+7 - Latency Breakdown & Speedup\n");
    printf("============================================================\n\n");

    printf("[1] Loading data... "); fflush(stdout);
    MMap sm; sm.load("all_spec_601d.bin");
    MMap l50; l50.load("lib_pca_50d.bin");
    MMap l150; l150.load("lib_pca_150d.bin");
    MMap l20; l20.load("lib_pca_20d.bin");

    vector<float> ltk(NMAT*NM);
    FILE* fp=fopen((string(DD)+"lib_thick.bin").c_str(),"rb"); fread(ltk.data(),4,NMAT*NM,fp); fclose(fp);

    MMap qsp[NL], q20[NL], q50[NL], q150[NL];
    for(int i=0;i<NL;i++){
        char buf[128];
        sprintf_s(buf,128,"noisy_q_%s_601d.bin",i==0?"clean":i==1?"1pct":i==2?"2pct":"5pct"); qsp[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_20d.bin",i==0?"clean":i==1?"1pct":i==2?"2pct":"5pct"); q20[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_50d.bin",i==0?"clean":i==1?"1pct":i==2?"2pct":"5pct"); q50[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_150d.bin",i==0?"clean":i==1?"1pct":i==2?"2pct":"5pct"); q150[i].load(buf);
    }
    vector<float> qtk(Q);
    fp=fopen((string(DD)+"q_gt.bin").c_str(),"rb"); fread(qtk.data(),4,Q,fp); fclose(fp);

    vector<float> lni; lni.resize(NMAT*NM);
    #pragma omp parallel for
    for(int i=0;i<NMAT*NM;i++){double s=0;for(int k=0;k<NDIM;k++){float v=sm.d[(size_t)i*NDIM+k];s+=v*v;}lni[i]=1.0f/(sqrtf((float)s)+1e-10f);}
    printf("ok\n\n"); fflush(stdout);

    printf("[2] Building KD-Trees... "); fflush(stdout);
    size_t NALL=(size_t)NMAT*NM;
    PC<20> rp20(l20.d,NALL); KDT<20> rt20(20,rp20,KDTreeSingleIndexAdaptorParams(30)); rt20.buildIndex();
    PC<50> tp50[NM]={PC<50>(l50.d,NMAT),PC<50>(l50.d+1ULL*NMAT*50,NMAT),PC<50>(l50.d+2ULL*NMAT*50,NMAT)};
    PC<150> tp150[NM]={PC<150>(l150.d,NMAT),PC<150>(l150.d+1ULL*NMAT*150,NMAT),PC<150>(l150.d+2ULL*NMAT*150,NMAT)};
    unique_ptr<KDT<50>> k50[NM]; unique_ptr<KDT<150>> k150[NM];
    for(int m=0;m<NM;m++){
        k50[m]=make_unique<KDT<50>>(50,tp50[m],KDTreeSingleIndexAdaptorParams(30)); k50[m]->buildIndex();
        k150[m]=make_unique<KDT<150>>(150,tp150[m],KDTreeSingleIndexAdaptorParams(30)); k150[m]->buildIndex();
    }
    printf("ok\n\n"); fflush(stdout);

    // Pre-compute routing for all noise levels
    int rm[NL][Q];
    for(int nl=0;nl<NL;nl++)
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt20.findNeighbors(rs,q20[nl].d+(size_t)qi*20);rm[nl][qi]=int(ni/NMAT);}

    // ================================================================
    //  PART 6: LATENCY BREAKDOWN (1% noise)
    // ================================================================
    printf("============================================================\n");
    printf(" PART 6: Latency Breakdown (1%% noise)\n");
    printf("============================================================\n\n");

    int nla=1;
    printf("%-35s %10s\n","Component","Lat/us");
    printf("------------------------------------------------------\n");

    // 6a. PCA-20D router 1-NN
    {
        Ti t;
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt20.findNeighbors(rs,q20[nla].d+(size_t)qi*20);(void)ni;}
        printf("%-35s %8.0f us\n","PCA-20D router (1-NN KDT)",t.us()/Q);
    }

    // 6b. KDT-50D search (oracle, single material)
    {
        Ti t;
        for(int qi=0;qi<Q;qi++){int m=qi/500;size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);(void)ni;}
        printf("%-35s %8.0f us\n","KDT-50D 1-NN (oracle)",t.us()/Q);
    }

    // 6c. KDT-150D search (oracle)
    {
        Ti t;
        for(int qi=0;qi<Q;qi++){int m=qi/500;size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);k150[m]->findNeighbors(rs,q150[nla].d+(size_t)qi*150);(void)ni;}
        printf("%-35s %8.0f us\n","KDT-150D 1-NN (oracle)",t.us()/Q);
    }

    // 6d. KDT-50D K=50 search (for rerank)
    {
        Ti t; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){int m=qi/500;KNNResultSet<float> rs(K);rs.init(idx,dst);k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);}
        printf("%-35s %8.0f us\n","KDT-50D K=50 (oracle)",t.us()/Q);
    }

    // 6e. Cosine rerank (K=50, AVX2)
    {
        Ti t; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            float bc=-2.0f;
            for(int ci=0;ci<K;ci++){float cs=avx_dot(sm.d+((size_t)m*NMAT+idx[ci])*NDIM,qq)*lni[(size_t)m*NMAT+idx[ci]]*qinv;if(cs>bc)bc=cs;}
        }
        printf("%-35s %8.0f us\n","Cosine rerank K=50 (AVX2)",t.us()/Q);
    }

    // 6f. BF-601D single material (OMP parallel)
    {
        Ti t; int dc=0;
        #pragma omp parallel for reduction(+:dc)
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            float bd=1e30f;
            for(size_t i=(size_t)m*NMAT;i<(size_t)(m+1)*NMAT;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd)bd=d;}
            dc+=int(bd);
        }
        printf("%-35s %8.0f us (%d threads)\n","BF-601D 500K (OMP parallel)",t.us()/Q,omp_get_max_threads()); (void)dc;
    }

    // 6g. BF-601D full 1.5M (OMP parallel)
    {
        Ti t; int dc=0;
        #pragma omp parallel for reduction(+:dc)
        for(int qi=0;qi<Q;qi++){
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            float bd=1e30f;
            for(size_t i=0;i<NALL;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd)bd=d;}
            dc+=int(bd);
        }
        printf("%-35s %8.0f us (full 1.5M, OMP)\n","BF-601D 1.5M (OMP parallel)",t.us()/Q); (void)dc;
    }

    // 6h. KDF full pipeline estimate (router+search+rerank)
    {
        Ti t; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=rm[nla][qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            float bc=-2.0f;
            for(int ci=0;ci<K;ci++){float cs=avx_dot(sm.d+((size_t)m*NMAT+idx[ci])*NDIM,qq)*lni[(size_t)m*NMAT+idx[ci]]*qinv;if(cs>bc)bc=cs;}
        }
        printf("%-35s %8.0f us\n","KDF-50/50 full pipeline",t.us()/Q);
    }
    printf("------------------------------------------------------\n\n"); fflush(stdout);

    // ================================================================
    //  PART 7: SPEEDUP SUMMARY (1% noise)
    // ================================================================
    printf("============================================================\n");
    printf(" PART 7: Speedup Summary vs BF-601D (1%% noise)\n");
    printf("============================================================\n\n");

    printf("%-30s %7s %9s %9s %7s\n","Method","P1nm","Lat/us","up/bf1.5M","Rout%");
    printf("----------------------------------------------------------------\n");

    float er[Q];
    // BF-601D full 1.5M (baseline)
    double bf_lat;
    {
        Ti t; int h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<Q;qi++){
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t best=0; float bd=1e30f;
            for(size_t i=0;i<NALL;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        bf_lat=t.us()/Q;
        printf("%-30s %5.1f%% %8.0f %7s %7.1f%%\n","BF-601D (1.5M)",100.0f*h/Q,bf_lat,"1.0x",100.0);
    }

    // KDF-50/50 with routing
    {
        Ti t; int h=0; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=rm[nla][qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[nla][qi]==qi/500;
        printf("%-30s %5.1f%% %8.0f %6.1fx %7.1f%%\n","20D+KDF-50/50",100.0f*h/Q,lat,bf_lat/lat,100.0f*rc/Q);
    }

    // KDT-50D with routing (no rerank)
    {
        Ti t; int h=0;
        for(int qi=0;qi<Q;qi++){
            int m=rm[nla][qi]; size_t ni; float nd;
            KNNResultSet<float> rs(1); rs.init(&ni,&nd);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[nla][qi]==qi/500;
        printf("%-30s %5.1f%% %8.0f %6.1fx %7.1f%%\n","20D+KDT-50D (no rerank)",100.0f*h/Q,lat,bf_lat/lat,100.0f*rc/Q);
    }

    // KDT-150D with routing
    {
        Ti t; int h=0;
        for(int qi=0;qi<Q;qi++){
            int m=rm[nla][qi]; size_t ni; float nd;
            KNNResultSet<float> rs(1); rs.init(&ni,&nd);
            k150[m]->findNeighbors(rs,q150[nla].d+(size_t)qi*150);
            er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[nla][qi]==qi/500;
        printf("%-30s %5.1f%% %8.0f %6.1fx %7.1f%%\n","20D+KDT-150D",100.0f*h/Q,lat,bf_lat/lat,100.0f*rc/Q);
    }

    // Oracle KDF-50/50
    {
        Ti t; int h=0; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        printf("%-30s %5.1f%% %8.0f %6.1fx %7.1f%%\n","Oracle KDF-50/50",100.0f*h/Q,t.us()/Q,bf_lat/(t.us()/Q),100.0);
    }

    // Oracle BF-601D (500K)
    {
        Ti t; int h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
            for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q;
        printf("%-30s %5.1f%% %8.0f %6.1fx %7.1f%%\n","Oracle BF-601D (500K)",100.0f*h/Q,lat,bf_lat/lat,100.0);
    }
    printf("----------------------------------------------------------------\n");

    // Cross-verify: re-run 20D+BF-601D and 20D+KDF-50/50 for 0%, 2%, 5% noise
    printf("\n");
    printf("============================================================\n");
    printf(" VERIFICATION: Cross-check with Part 2 data\n");
    printf("============================================================\n\n");

    for(int nl=0;nl<NL;nl++){
        printf("[Noise %s]\n",QFN[nl]);
        printf("%-30s %7s\n","Method","P1nm");
        printf("--------------------------------------------\n");
        // BF-601D with routing (OMP)
        {
            Ti t; int h=0;
            #pragma omp parallel for reduction(+:h)
            for(int qi=0;qi<Q;qi++){
                int m=rm[nl][qi]; const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
                for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            printf("%-30s %5.1f%% (ref %5.1f%%)\n","20D+BF-601D",100.0f*h/Q,
                nl==0?100.0:nl==1?99.3:nl==2?98.7:91.0);
        }
        // KDF-50/50 with routing
        {
            Ti t; int h=0; const int K=50; size_t idx[K]; float dst[K];
            for(int qi=0;qi<Q;qi++){
                int m=rm[nl][qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
                size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
                for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            printf("%-30s %5.1f%% (ref %5.1f%%)\n","20D+KDF-50/50",100.0f*h/Q,
                nl==0?100.0:nl==1?99.3:nl==2?98.7:86.5);
        }
        printf("--------------------------------------------\n");
    }

    printf("\n============================================================\n");
    printf(" PART 6+7 COMPLETE (OOM-free version)\n");
    printf("============================================================\n");
    return 0;
}

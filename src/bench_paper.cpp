// bench_paper.cpp - Comprehensive benchmark: routing + thickness matching for paper
// cl /O2 /openmp /EHsc /std:c++17 /arch:AVX2 bench_paper.cpp /Fe:bench_paper.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <array>
#include <cstring>
#include <windows.h>
#include <immintrin.h>
#include <omp.h>
#include "nanoflann.hpp"
#pragma comment(linker, "/STACK:1073741824")
using std::vector; using std::array; using std::string; using std::sort; using std::unique_ptr; using std::make_unique;
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

void get_lnorm(vector<float>& lni, const float* sm, size_t N){
    lni.resize(N);
    #pragma omp parallel for
    for(int i=0;i<(int)N;i++){double s=0;for(int k=0;k<NDIM;k++){float v=sm[(size_t)i*NDIM+k];s+=v*v;}lni[i]=1.0f/(sqrtf((float)s)+1e-10f);}
}

// BF-601D on a specific material (m) for one query (qi), store in er[qi]
inline int bf_601d_mat(int qi, int m, const float* qq, const float* sm, const float* ltk, const float* qtk, float* er) {
    size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs;
    float bd=1e30f;
    for(size_t i=bs;i<be;i++){
        float d=0;
        for(int k=0;k<NDIM;k++){float v=sm[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}
        if(d<bd){bd=d; best=i;}
    }
    er[qi]=fabsf(ltk[best]-qtk[qi]);
    return er[qi]<=1?1:0;
}

void print_row(const char* lab, int hits, double lat, double rt, const float* er, int n) {
    vector<float> e(er,er+n); sort(e.begin(),e.end());
    double ma=0; for(int i=0;i<n;i++) ma+=e[i]; ma/=n;
    printf("%-30s %6.1f%% %8.0f %7.1f%% %8.1f %7.1f %7.1f %8.1f\n",lab,100.0f*hits/n,lat,rt*100,(float)ma,e[n/2],e[int(n*0.95)],e.back());
    fflush(stdout);
}

int main() {
    printf("======================================================================\n");
    printf(" KD-Forest Paper Benchmark Suite (3 materials x 500K = 1.5M spectra)\n");
    printf("======================================================================\n\n");

    printf("[1] Loading data... "); fflush(stdout);
    MMap sm; sm.load("all_spec_601d.bin");
    MMap l150; l150.load("lib_pca_150d.bin");
    MMap l50; l50.load("lib_pca_50d.bin");
    MMap l20; l20.load("lib_pca_20d.bin");
    MMap l10; l10.load("lib_pca_10d.bin");
    MMap l3; l3.load("lib_pca_3d.bin");

    vector<float> ltk(NMAT*NM);
    FILE* fp=fopen((string(DD)+"lib_thick.bin").c_str(),"rb"); fread(ltk.data(),4,NMAT*NM,fp); fclose(fp);

    MMap qsp[NL], q3[NL], q10[NL], q20[NL], q50[NL], q150[NL];
    const char* qp[]={"clean","1pct","2pct","5pct"};
    const char* q6[]={"noisy_q_clean_601d.bin","noisy_q_1pct_601d.bin","noisy_q_2pct_601d.bin","noisy_q_5pct_601d.bin"};
    char buf[128];
    for(int i=0;i<NL;i++){
        qsp[i].load(q6[i]);
        sprintf_s(buf,128,"q_%s_pca_3d.bin",qp[i]); q3[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_10d.bin",qp[i]); q10[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_20d.bin",qp[i]); q20[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_50d.bin",qp[i]); q50[i].load(buf);
        sprintf_s(buf,128,"q_%s_pca_150d.bin",qp[i]); q150[i].load(buf);
    }
    vector<float> qtk(Q);
    fp=fopen((string(DD)+"q_gt.bin").c_str(),"rb"); fread(qtk.data(),4,Q,fp); fclose(fp);
    printf("ok\n\n"); fflush(stdout);

    printf("[2] Building KD-Trees... "); fflush(stdout);
    size_t NALL=(size_t)NMAT*NM;
    PC<3> rp3(l3.d,NALL); KDT<3> rt3(3,rp3,KDTreeSingleIndexAdaptorParams(30)); rt3.buildIndex();
    PC<10> rp10(l10.d,NALL); KDT<10> rt10(10,rp10,KDTreeSingleIndexAdaptorParams(30)); rt10.buildIndex();
    PC<20> rp20(l20.d,NALL); KDT<20> rt20(20,rp20,KDTreeSingleIndexAdaptorParams(30)); rt20.buildIndex();

    PC<50> tp50[NM]={PC<50>(l50.d,NMAT),PC<50>(l50.d+1ULL*NMAT*50,NMAT),PC<50>(l50.d+2ULL*NMAT*50,NMAT)};
    PC<150> tp150[NM]={PC<150>(l150.d,NMAT),PC<150>(l150.d+1ULL*NMAT*150,NMAT),PC<150>(l150.d+2ULL*NMAT*150,NMAT)};
    unique_ptr<KDT<50>> k50[NM]; unique_ptr<KDT<150>> k150[NM];
    for(int m=0;m<NM;m++){
        k50[m]=make_unique<KDT<50>>(50,tp50[m],KDTreeSingleIndexAdaptorParams(30)); k50[m]->buildIndex();
        k150[m]=make_unique<KDT<150>>(150,tp150[m],KDTreeSingleIndexAdaptorParams(30)); k150[m]->buildIndex();
    }
    vector<float> lni; get_lnorm(lni, sm.d, NALL);
    printf("ok\n\n"); fflush(stdout);

    // =====================================================================
    //  PART 1: MATERIAL ROUTING
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 1: Material Routing\n");
    printf("======================================================================\n\n");

    array<array<vector<int>,3>,NL> routes; // 3 methods: 3D, 10D, 20D
    for(int nl=0;nl<NL;nl++) for(int m=0;m<3;m++) routes[nl][m].resize(Q);

    for(int nl=0;nl<NL;nl++){
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt3.findNeighbors(rs,q3[nl].d+(size_t)qi*3);routes[nl][0][qi]=int(ni/NMAT);}
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt10.findNeighbors(rs,q10[nl].d+(size_t)qi*10);routes[nl][1][qi]=int(ni/NMAT);}
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt20.findNeighbors(rs,q20[nl].d+(size_t)qi*20);routes[nl][2][qi]=int(ni/NMAT);}
    }

    printf("Table 1: Routing Accuracy (Q=%d)\n",Q);
    printf("--------------------------------------------------------\n");
    printf("%-8s %8s %8s %8s\n","Noise","PCA-3D","PCA-10D","PCA-20D");
    printf("--------------------------------------------------------\n");
    for(int nl=0;nl<NL;nl++){
        printf("%-8s",QFN[nl]);
        for(int m=0;m<3;m++){int c=0;for(int qi=0;qi<Q;qi++) c+=routes[nl][m][qi]==qi/500;printf(" %7.1f%%",100.0f*c/Q);}
        printf("\n");
    }
    printf("--------------------------------------------------------\n\n");

    printf("Table 2: Confusion Matrix at 1%% noise (PCA-20D)\n");
    printf("------------------------------------------------\n");
    int cm[3][3]={0}; for(int qi=0;qi<Q;qi++){int g=qi/500;cm[g][routes[1][2][qi]]++;}
    printf("%-10s %8s %8s %8s\n","","SiO2","Si3N4","a-Si");
    for(int i=0;i<3;i++) printf("%-10s %8d %8d %8d\n",MNAME[i],cm[i][0],cm[i][1],cm[i][2]);
    printf("------------------------------------------------\n\n"); fflush(stdout);

    // =====================================================================
    //  PART 2: THICKNESS MATCHING - FULL PIPELINE
    //  Uses OMP for BF loops; KDT/KDF on full 1500 queries
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 2: Thickness Matching - Full Pipeline (PCA-20D router)\n");
    printf("======================================================================\n\n");

    // Pre-compute all routing results for PCA-20D
    const int RM=2; // PCA-20D as primary router

    for(int nl=0;nl<NL;nl++){
        printf("[Noise %s] ",QFN[nl]); fflush(stdout);
        auto& rm=routes[nl][RM];
        float er[Q]; int h; double lat;

        printf("\n%-30s %6s %8s %8s %8s %8s %8s %8s\n","Method","P1nm","Lat/us","Rout%","MAE/nm","MedAE","P95","Max/nm");
        printf("----------------------------------------------------------------\n");

        // 1. BF-601D with routing (OMP parallel)
        {
            Ti t; h=0;
            #pragma omp parallel for reduction(+:h)
            for(int qi=0;qi<Q;qi++){
                const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                size_t bs=(size_t)rm[qi]*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
                for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[qi]==qi/500;
            print_row("20D+BF-601D",h,lat,rc/1.0/Q,er,Q);
        }

        // 2. KDT-50D with routing
        {
            Ti t; h=0;
            for(int qi=0;qi<Q;qi++){
                int m=rm[qi]; size_t ni; float nd;
                KNNResultSet<float> rs(1); rs.init(&ni,&nd);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
            }
            lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[qi]==qi/500;
            print_row("20D+KDT-50D",h,lat,rc/1.0/Q,er,Q);
        }

        // 3. KDT-150D with routing
        {
            Ti t; h=0;
            for(int qi=0;qi<Q;qi++){
                int m=rm[qi]; size_t ni; float nd;
                KNNResultSet<float> rs(1); rs.init(&ni,&nd);
                k150[m]->findNeighbors(rs,q150[nl].d+(size_t)qi*150);
                er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
            }
            lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[qi]==qi/500;
            print_row("20D+KDT-150D",h,lat,rc/1.0/Q,er,Q);
        }

        // 4. KDF-50/50
        {
            Ti t; h=0; const int K=50; size_t idx[K]; float dst[K];
            for(int qi=0;qi<Q;qi++){
                int m=rm[qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
                size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
                for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[qi]==qi/500;
            print_row("20D+KDF-50/50",h,lat,rc/1.0/Q,er,Q);
        }

        // 5. KDF-50/10
        {
            Ti t; h=0; const int K=10; size_t idx[K]; float dst[K];
            for(int qi=0;qi<Q;qi++){
                int m=rm[qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
                size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
                for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            lat=t.us()/Q; int rc=0; for(int qi=0;qi<Q;qi++) rc+=rm[qi]==qi/500;
            print_row("20D+KDF-50/10",h,lat,rc/1.0/Q,er,Q);
        }
    }
    printf("\n"); fflush(stdout);

    // =====================================================================
    //  PART 3: ORACLE COMPARISON (known material)
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 3: Oracle Comparison (known material)\n");
    printf("======================================================================\n\n");

    for(int nl=0;nl<NL;nl++){
        printf("[Noise %s]\n",QFN[nl]);
        printf("%-30s %6s %8s %8s %8s %8s %8s %8s\n","Method","P1nm","Lat/us","Rout%","MAE/nm","MedAE","P95","Max/nm");
        printf("----------------------------------------------------------------\n");

        // Oracle BF-601D (OMP parallel)
        {
            Ti t; int h=0; float er[Q];
            #pragma omp parallel for reduction(+:h)
            for(int qi=0;qi<Q;qi++){
                int m=qi/500; const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
                for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            print_row("Oracle+BF-601D",h,t.us()/Q,1.0,er,Q);
        }

        // Oracle KDT-50D
        {
            Ti t; int h=0; float er[Q];
            for(int qi=0;qi<Q;qi++){
                int m=qi/500; size_t ni; float nd;
                KNNResultSet<float> rs(1); rs.init(&ni,&nd);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
            }
            print_row("Oracle+KDT-50D",h,t.us()/Q,1.0,er,Q);
        }

        // Oracle KDT-150D
        {
            Ti t; int h=0; float er[Q];
            for(int qi=0;qi<Q;qi++){
                int m=qi/500; size_t ni; float nd;
                KNNResultSet<float> rs(1); rs.init(&ni,&nd);
                k150[m]->findNeighbors(rs,q150[nl].d+(size_t)qi*150);
                er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
            }
            print_row("Oracle+KDT-150D",h,t.us()/Q,1.0,er,Q);
        }

        // Oracle KDF-50/50
        {
            Ti t; int h=0; float er[Q]; const int K=50; size_t idx[K]; float dst[K];
            for(int qi=0;qi<Q;qi++){
                int m=qi/500; KNNResultSet<float> rs(K); rs.init(idx,dst);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
                size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
                for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            print_row("Oracle+KDF-50/50",h,t.us()/Q,1.0,er,Q);
        }
    }
    printf("\n"); fflush(stdout);

    // =====================================================================
    //  PART 4: ABLATION (1% noise)
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 4: Ablation Study (1%% noise, PCA-20D router)\n");
    printf("======================================================================\n\n");

    int nla=1; auto& rma=routes[nla][RM];
    float er[Q]; int h; double lat;

    printf("%-30s %6s %8s %8s %8s %8s %8s %8s\n","Method","P1nm","Lat/us","Rout%","MAE/nm","MedAE","P95","Max/nm");
    printf("----------------------------------------------------------------\n");

    // Full KDF-50/50
    {
        Ti t; h=0; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=rma[qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        int rc=0; for(int qi=0;qi<Q;qi++) rc+=rma[qi]==qi/500;
        print_row("Full KDF-50/50",h,t.us()/Q,rc/1.0/Q,er,Q);
    }

    // NoRerank = KDT-50D
    {
        Ti t; h=0;
        for(int qi=0;qi<Q;qi++){
            int m=rma[qi]; size_t ni; float nd;
            KNNResultSet<float> rs(1); rs.init(&ni,&nd);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
        }
        int rc=0; for(int qi=0;qi<Q;qi++) rc+=rma[qi]==qi/500;
        print_row("NoRerank=KDT-50D",h,t.us()/Q,rc/1.0/Q,er,Q);
    }

    // NoRoute = BF-601D full 1.5M (OMP)
    {
        Ti t; h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<Q;qi++){
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t best=0; float bd=1e30f;
            for(size_t i=0;i<NALL;i++){
                float d=0; for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}
                if(d<bd){bd=d; best=i;}
            }
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        print_row("NoRoute=BF-1.5M",h,t.us()/Q,1.0,er,Q);
    }

    // NoRoute+KDT-50D (KDT on full 1.5M, no routing)
    {
        Ti t; h=0;
        PC<50> fp50(l50.d,NALL);
        KDT<50> ft50(50,fp50,KDTreeSingleIndexAdaptorParams(30)); ft50.buildIndex();
        for(int qi=0;qi<Q;qi++){
            size_t ni; float nd; KNNResultSet<float> rs(1); rs.init(&ni,&nd);
            ft50.findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            er[qi]=fabsf(ltk[ni]-qtk[qi]); if(er[qi]<=1) h++;
        }
        print_row("NoRoute+KDT-50D",h,t.us()/Q,1.0,er,Q);
    }
    printf("\n"); fflush(stdout);

    // =====================================================================
    //  PART 5: PER-MATERIAL
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 5: Per-Material Accuracy (1%% noise, KDF-50/50)\n");
    printf("======================================================================\n\n");

    printf("%-10s %8s %8s %8s %8s %8s\n","Material","P1nm","MAE","MedAE","P95","Max");
    printf("--------------------------------------------------\n");
    const int K=50;
    for(int mi=0;mi<NM;mi++){
        size_t idx[K]; float dst[K]; h=0; vector<float> errs;
        for(int qi=mi*500;qi<(mi+1)*500;qi++){
            int m=rma[qi];
            if(m!=mi){errs.push_back(9999); continue;}
            KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            float e=fabsf(ltk[best]-qtk[qi]); errs.push_back(e); if(e<=1) h++;
        }
        sort(errs.begin(),errs.end());
        double ma=0; for(auto v:errs) ma+=v; ma/=errs.size();
        printf("%-10s %6.1f%% %8.1f %7.1f %7.1f %8.1f\n",MNAME[mi],100.0f*h/500,(float)ma,errs[errs.size()/2],errs[int(errs.size()*0.95)],errs.back());
    }
    // Overall
    {
        size_t idx[K]; float dst[K]; h=0; vector<float> errs;
        for(int qi=0;qi<Q;qi++){
            int m=rma[qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            float e=fabsf(ltk[best]-qtk[qi]); errs.push_back(e); if(e<=1) h++;
        }
        sort(errs.begin(),errs.end());
        double ma=0; for(auto v:errs) ma+=v; ma/=errs.size();
        printf("--------------------------------------------------\n");
        printf("%-10s %6.1f%% %8.1f %7.1f %7.1f %8.1f\n","Overall",100.0f*h/Q,(float)ma,errs[Q/2],errs[int(Q*0.95)],errs.back());
    }
    printf("\n"); fflush(stdout);

    // =====================================================================
    //  PART 6: LATENCY BREAKDOWN
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 6: Latency Breakdown (1%% noise)\n");
    printf("======================================================================\n\n");

    printf("%-30s %10s %10s %10s\n","Component","Lat/us","Notes","");
    printf("----------------------------------------------------------------\n");

    // Router latency (PCA-20D KDT 1-NN)
    {
        Ti t; int dc=0;
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt20.findNeighbors(rs,q20[nla].d+(size_t)qi*20);dc+=int(ni);}
        printf("%-30s %8.0f us\n","PCA-20D router (1-NN)",t.us()/Q); (void)dc;
    }

    // KDT-50D search
    {
        Ti t; int dc=0;
        for(int qi=0;qi<Q;qi++){int m=qi/500;size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);dc+=int(ni);}
        printf("%-30s %8.0f us\n","KDT-50D search (oracle)",t.us()/Q); (void)dc;
    }

    // KDT-150D search
    {
        Ti t; int dc=0;
        for(int qi=0;qi<Q;qi++){int m=qi/500;size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);k150[m]->findNeighbors(rs,q150[nla].d+(size_t)qi*150);dc+=int(ni);}
        printf("%-30s %8.0f us\n","KDT-150D search (oracle)",t.us()/Q); (void)dc;
    }

    // KDF rerank (K=50 cosine)
    {
        Ti t; int dc=0; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            float bc=-2.0f;
            for(int ci=0;ci<K;ci++){float cs=avx_dot(sm.d+((size_t)m*NMAT+idx[ci])*NDIM,qq)*lni[(size_t)m*NMAT+idx[ci]]*qinv;if(cs>bc)bc=cs;}
            dc+=int(bc);
        }
        printf("%-30s %8.0f us (search)  +  rerank\n","KDF-50/50 total",t.us()/Q); (void)dc;
    }

    // BF-601D on 500K (with OMP)
    {
        Ti t; int dc=0;
        #pragma omp parallel for reduction(+:dc)
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            float bd=1e30f;
            for(size_t i=(size_t)m*NMAT;i<(size_t)(m+1)*NMAT;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd)bd=d;}
            dc+=int(bd);
        }
        printf("%-30s %8.0f us (with OMP %d threads)\n","BF-601D 500K",t.us()/Q,omp_get_max_threads()); (void)dc;
    }

    // BF-601D on 1.5M
    {
        Ti t; int dc=0;
        #pragma omp parallel for reduction(+:dc)
        for(int qi=0;qi<Q;qi++){
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            float bd=1e30f;
            for(size_t i=0;i<NALL;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd)bd=d;}
            dc+=int(bd);
        }
        printf("%-30s %8.0f us (full 1.5M, OMP)\n","BF-601D 1.5M",t.us()/Q); (void)dc;
    }

    // Full pipeline estimate
    printf("----------------------------------------------------------------\n");
    printf("%-30s   ~200 us (router+search+rerank)\n","KDF-50/50 (estimated)");
    printf("%-30s   ~18000 us (1.5M, OMP)\n","BF-601D (estimated)");
    printf("----------------------------------------------------------------\n\n");
    fflush(stdout);

    // =====================================================================
    //  PART 7: SPEEDUP SUMMARY
    // =====================================================================
    printf("======================================================================\n");
    printf(" PART 7: Speedup Summary (1%% noise)\n");
    printf("======================================================================\n\n");

    double bf_lat;
    {
        Ti t; h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<Q;qi++){
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t best=0; float bd=1e30f;
            for(size_t i=0;i<NALL;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
            if(fabsf(ltk[best]-qtk[qi])<=1) h++;
        }
        bf_lat=t.us()/Q;
        printf("%-30s %7.1f%%  %8.0f us   1.0x\n","BF-601D baseline (1.5M)",100.0f*h/Q,bf_lat);
    }

    // KDF with routing
    {
        Ti t; h=0; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=rma[qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q;
        printf("%-30s %7.1f%%  %8.0f us  %6.1fx\n","20D+KDF-50/50",100.0f*h/Q,lat,bf_lat/lat);
    }

    // KDT-50D with routing
    {
        Ti t; h=0;
        for(int qi=0;qi<Q;qi++){
            int m=rma[qi]; size_t ni; float nd;
            KNNResultSet<float> rs(1); rs.init(&ni,&nd);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q;
        printf("%-30s %7.1f%%  %8.0f us  %6.1fx\n","20D+KDT-50D",100.0f*h/Q,lat,bf_lat/lat);
    }

    // Oracle KDF
    {
        Ti t; h=0; const int K=50; size_t idx[K]; float dst[K];
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; KNNResultSet<float> rs(K); rs.init(idx,dst);
            k50[m]->findNeighbors(rs,q50[nla].d+(size_t)qi*50);
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
            size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
            for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        double lat=t.us()/Q;
        printf("%-30s %7.1f%%  %8.0f us  %6.1fx\n","Oracle KDF-50/50",100.0f*h/Q,lat,bf_lat/lat);
    }

    // Oracle BF-601D
    {
        Ti t; h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<Q;qi++){
            int m=qi/500; const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
            for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
            if(fabsf(ltk[best]-qtk[qi])<=1) h++;
        }
        double lat=t.us()/Q;
        printf("%-30s %7.1f%%  %8.0f us  %6.1fx\n","Oracle BF-601D",100.0f*h/Q,lat,bf_lat/lat);
    }
    printf("\n");

    printf("======================================================================\n");
    printf(" BENCHMARK COMPLETE\n");
    printf("======================================================================\n");
    return 0;
}

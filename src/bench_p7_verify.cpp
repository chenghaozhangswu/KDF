// bench_p7_verify.cpp - Part 7 Speedup + Cross-verify with subsampling for BF loops
// Uses 150 queries for BF-heavy sections, full 1500 for KDT/KDF
// cl /O2 /openmp /EHsc /std:c++17 /arch:AVX2 bench_p7_verify.cpp /Fe:p7verify.exe
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
enum {NDIM=601, NMAT=500000, Q=1500, QBF=150, NL=4, NM=3};
const char* MNAME[]={"SiO2","Si3N4","a-Si"}; // not used in final
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

void print_row(const char* lab, int hits, int n, double lat, double rt, const float* er) {
    vector<float> e(er,er+n); sort(e.begin(),e.end());
    double ma=0; for(int i=0;i<n;i++) ma+=e[i]; ma/=n;
    printf("%-30s %6.1f%% %8.0f %7.1f%% %8.1f %7.1f %7.1f %8.1f\n",lab,100.0f*hits/n,lat,rt*100,(float)ma,e[n/2],e[int(n*0.95)],e.back());
}

int main() {
    printf("============================================================\n");
    printf("P7+Verify: Speedup Summary & Cross-Verification\n");
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

    vector<float> lni(NMAT*NM);
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

    // Route table
    int rm[NL][Q];
    for(int nl=0;nl<NL;nl++)
        for(int qi=0;qi<Q;qi++){size_t ni;float nd;KNNResultSet<float> rs(1);rs.init(&ni,&nd);rt20.findNeighbors(rs,q20[nl].d+(size_t)qi*20);rm[nl][qi]=int(ni/NMAT);}

    int nla=1; // 1% noise
    float er[Q];

    // ================================================================
    // PART 7: SPEEDUP SUMMARY (1% noise, BF-LOOPS use QBF=150 queries)
    // ================================================================
    printf("============================================================\n");
    printf(" PART 7: Speedup Summary vs BF-601D (1%% noise)\n");
    printf("============================================================\n");
    printf("(BF-heavy loops use %d queries; KDT/KDF use full %d)\n\n", QBF, Q);

    printf("%-30s %6s %8s %8s %8s %8s %8s %8s\n","Method","P1nm","Lat/us","Rout%","MAE/nm","MedAE","P95","Max/nm");
    printf("----------------------------------------------------------------\n");

    // Baseline: BF-601D 1.5M (QBF queries, OMP)
    double bf_lat;
    {
        Ti t; int h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<QBF;qi++){
            const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t best=0; float bd=1e30f;
            for(size_t i=0;i<NALL;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        bf_lat=t.us()/QBF;
        printf("%-30s %5.1f%% %8.0f %7s %8.1f %7.1f %7.1f %8.1f\n","BF-601D 1.5M",100.0f*h/QBF,bf_lat,"1.0x",(float)0.0,(float)0.0,(float)0.0,(float)0.0);
        // Full stats for baseline
        {
            float ebf[QBF]; for(int qi=0;qi<QBF;qi++)ebf[qi]=er[qi];
            sort(ebf,ebf+QBF); double ma=0; for(int i=0;i<QBF;i++)ma+=ebf[i];ma/=QBF;
            printf("  (details)       P1nm=%5.1f%% MAE=%.1f MedAE=%.1f P95=%.1f Max=%.1f\n",
                100.0f*h/QBF,(float)ma,ebf[QBF/2],ebf[int(QBF*0.95)],ebf[QBF-1]);
        }
    }

    // KDF-50/50 with routing (full Q)
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
        print_row("PCA20D+KDF-50/50",h,Q,lat,rc/1.0/Q,er);
        printf("  Speedup: %.0fx (BF %.0fus / KDF %.0fus)\n",bf_lat/lat,bf_lat,lat);
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
        print_row("PCA20D+KDT-50D (no rerank)",h,Q,lat,rc/1.0/Q,er);
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
        print_row("PCA20D+KDT-150D",h,Q,lat,rc/1.0/Q,er);
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
        print_row("Oracle KDF-50/50",h,Q,t.us()/Q,1.0,er);
    }

    // Oracle BF-601D (500K, QBF queries, OMP)
    {
        Ti t; int h=0;
        #pragma omp parallel for reduction(+:h)
        for(int qi=0;qi<QBF;qi++){
            int m=qi/500; const float* qq=qsp[nla].d+(size_t)qi*NDIM;
            size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
            for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
            er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
        }
        print_row("Oracle BF-601D (500K)",h,QBF,t.us()/QBF,1.0,er);
    }
    printf("----------------------------------------------------------------\n\n"); fflush(stdout);

    // ================================================================
    // VERIFICATION: Cross-check with Part 2 data
    // ================================================================
    printf("============================================================\n");
    printf(" CROSS-VERIFICATION: Re-run key Part 2 results\n");
    printf("============================================================\n\n");

    // Reference values from bench_paper_v2.exe Part 2
    const double P2_REF[4][5] = {
        {100.0, 100.0, 100.0, 100.0, 100.0},  // 0% noise: BF, KDT50, KDT150, KDF50/50, KDF50/10
        {99.3, 91.9, 99.3, 99.3, 99.1},        // 1%
        {98.7, 83.3, 98.7, 98.7, 96.5},        // 2%
        {91.0, 65.8, 89.9, 86.5, 78.5}         // 5%
    };
    const char* MLAB[]={"BF-601D","KDT-50D","KDT-150D","KDF-50/50","KDF-50/10"};

    for(int nl=0;nl<NL;nl++){
        printf("[Noise %s]  (Q=%d for BF, %d for KDT/KDF)\n",QFN[nl],QBF,Q);
        printf("%-30s %7s %8s %s\n","Method","P1nm","Lat/us","Ref check");
        printf("------------------------------------------------------\n");
        // BF-601D (QBF)
        {
            Ti t; int h=0;
            #pragma omp parallel for reduction(+:h)
            for(int qi=0;qi<QBF;qi++){
                int m=rm[nl][qi]; const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                size_t bs=(size_t)m*NMAT, be=bs+NMAT, best=bs; float bd=1e30f;
                for(size_t i=bs;i<be;i++){float d=0;for(int k=0;k<NDIM;k++){float v=sm.d[i*NDIM+k];d+=(v-qq[k])*(v-qq[k]);}if(d<bd){bd=d;best=i;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            double p=100.0f*h/QBF; double dr=p-P2_REF[nl][0];
            printf("%-30s %5.1f%% %8.0f (ref %.1f%%, diff %+.1f%%)\n","PCA20D+BF-601D",p,t.us()/QBF,P2_REF[nl][0],dr);
        }
        // KDF-50/50 (full Q)
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
            double p=100.0f*h/Q; double dr=p-P2_REF[nl][3];
            printf("%-30s %5.1f%% %8.0f (ref %.1f%%, diff %+.1f%%)\n","PCA20D+KDF-50/50",p,t.us()/Q,P2_REF[nl][3],dr);
        }
        // KDT-50D (full Q)
        {
            Ti t; int h=0;
            for(int qi=0;qi<Q;qi++){
                int m=rm[nl][qi]; size_t ni; float nd;
                KNNResultSet<float> rs(1); rs.init(&ni,&nd);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
            }
            double p=100.0f*h/Q; double dr=p-P2_REF[nl][1];
            printf("%-30s %5.1f%% %8.0f (ref %.1f%%, diff %+.1f%%)\n","PCA20D+KDT-50D",p,t.us()/Q,P2_REF[nl][1],dr);
        }
        // KDT-150D (full Q)
        {
            Ti t; int h=0;
            for(int qi=0;qi<Q;qi++){
                int m=rm[nl][qi]; size_t ni; float nd;
                KNNResultSet<float> rs(1); rs.init(&ni,&nd);
                k150[m]->findNeighbors(rs,q150[nl].d+(size_t)qi*150);
                er[qi]=fabsf(ltk[(size_t)m*NMAT+ni]-qtk[qi]); if(er[qi]<=1) h++;
            }
            double p=100.0f*h/Q; double dr=p-P2_REF[nl][2];
            printf("%-30s %5.1f%% %8.0f (ref %.1f%%, diff %+.1f%%)\n","PCA20D+KDT-150D",p,t.us()/Q,P2_REF[nl][2],dr);
        }
        // KDF-50/10 (full Q)
        {
            Ti t; int h=0; const int K=10; size_t idx[K]; float dst[K];
            for(int qi=0;qi<Q;qi++){
                int m=rm[nl][qi]; KNNResultSet<float> rs(K); rs.init(idx,dst);
                k50[m]->findNeighbors(rs,q50[nl].d+(size_t)qi*50);
                const float* qq=qsp[nl].d+(size_t)qi*NDIM;
                double qn=0; for(int k=0;k<NDIM;k++) qn+=qq[k]*qq[k]; float qinv=1.0f/(sqrtf((float)qn)+1e-10f);
                size_t best=(size_t)m*NMAT+idx[0]; float bc=-2.0f;
                for(int ci=0;ci<K;ci++){size_t ai=(size_t)m*NMAT+idx[ci];float cs=avx_dot(sm.d+ai*NDIM,qq)*lni[ai]*qinv;if(cs>bc){bc=cs;best=ai;}}
                er[qi]=fabsf(ltk[best]-qtk[qi]); if(er[qi]<=1) h++;
            }
            double p=100.0f*h/Q; double dr=p-P2_REF[nl][4];
            printf("%-30s %5.1f%% %8.0f (ref %.1f%%, diff %+.1f%%)\n","PCA20D+KDF-50/10",p,t.us()/Q,P2_REF[nl][4],dr);
        }
        printf("------------------------------------------------------\n\n",QFN[nl]);
        fflush(stdout);
    }

    printf("============================================================\n");
    printf(" ALL DONE\n");
    printf("============================================================\n");
    return 0;
}

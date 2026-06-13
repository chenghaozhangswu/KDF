// bench_coarse_fine.cpp - adaptive candidates via reconstruction error
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <algorithm>
#include <random>
#include <immintrin.h>
#include <nanoflann.hpp>
using namespace std;
using f32 = float;
constexpr int N=601,NMAT=4,NQ=500,DCOARSE=10,NCAND_BASE=10,NCAND_NOISE=200;
const f32 RECON_THRESH=0.02f;
const char*MNP[]={"ox","sin","soi","cauthy"};
const int SZV[]={10000,50000,100000,500000};
vector<f32>loadf(const string&p){
    FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"FAIL: %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END);size_t sz=ftell(f);fseek(f,0,SEEK_SET);
    vector<f32>b(sz/4);fread(b.data(),4,b.size(),f);fclose(f);return b;
}
inline f32 d2(const f32*a,const f32*b,int n){
    __m256 s=_mm256_setzero_ps();int k=0;
    for(;k+8<=n;k+=8){__m256 va=_mm256_loadu_ps(a+k),vb=_mm256_loadu_ps(b+k),d=_mm256_sub_ps(va,vb);
    s=_mm256_add_ps(s,_mm256_mul_ps(d,d));}
    __m128 lo=_mm256_castps256_ps128(s),hi=_mm256_extractf128_ps(s,1);
    __m128 ss=_mm_add_ps(lo,hi);ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);
    f32 r=_mm_cvtss_f32(ss);for(;k<n;k++){f32 d=a[k]-b[k];r+=d*d;}return r;
}
struct PC10{vector<f32>pts;size_t kdtree_get_point_count()const{return pts.size()/DCOARSE;}
    f32 kdtree_get_pt(size_t i,size_t d)const{return pts[i*DCOARSE+d];}
    template<class B>bool kdtree_get_bbox(B&)const{return false;}};
using KT10=nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<f32,PC10>,PC10,DCOARSE>;

void pcproj(const f32*r,f32*o,const f32*m,const f32*c,int nd){
    for(int d=0;d<nd;d++){const f32*cp=c+d*N;__m256 s=_mm256_setzero_ps();
    for(int j=0;j<=592;j+=8){__m256 rd=_mm256_sub_ps(_mm256_loadu_ps(r+j),_mm256_loadu_ps(m+j));
    s=_mm256_fmadd_ps(rd,_mm256_loadu_ps(cp+j),s);}
    f32 bf[8];_mm256_storeu_ps(bf,s);f32 t=bf[0]+bf[1]+bf[2]+bf[3]+bf[4]+bf[5]+bf[6]+bf[7];
    for(int j=600;j<N;j++)t+=(r[j]-m[j])*cp[j];o[d]=t;}
}

int main(){
    setbuf(stdout,NULL);
    string BP="D:\\\\kd_forest_v2_gh\\\\src\\\\multi\\\\";
    printf("Adaptive: 10D PCA + recon-threshold candidate selection\\n");
    for(int si=0;si<4;si++){
        int NR=SZV[si],NTOT=NR*NMAT;string SN=si==0?"10k":si==1?"50k":si==2?"100k":"500k";
        printf("\\n=== %s (NR=%d) ===\\n",SN.c_str(),NR);
        auto gm=loadf(BP+"gmean_"+SN+".bin");auto gc50=loadf(BP+"gcomp50_"+SN+".bin");
        vector<f32>gc10(DCOARSE*N);for(int d=0;d<DCOARSE;d++)memcpy(&gc10[d*N],&gc50[d*N],N*4);
        vector<vector<f32>>ms(NMAT),mt(NMAT);
        for(int m=0;m<NMAT;m++){ms[m]=loadf(BP+"lib_"+MNP[m]+"_n_"+SN+".bin");
            mt[m]=loadf(BP+"lib_"+MNP[m]+"_thick_"+SN+".bin");}
        vector<f32>as(NTOT*N),at(NTOT);
        for(int m=0;m<NMAT;m++){memcpy(&as[m*NR*N],ms[m].data(),NR*N*4);memcpy(&at[m*NR],mt[m].data(),NR*4);}
        // 10D PCA + KDT
        vector<f32>p10(NTOT*DCOARSE);
        for(int i=0;i<NTOT;i++)pcproj(&as[i*N],&p10[i*DCOARSE],gm.data(),gc10.data(),DCOARSE);
        PC10 p10d{p10};KT10 kdt10(DCOARSE,p10d,nanoflann::KDTreeSingleIndexAdaptorParams(30));
        kdt10.buildIndex();
        // queries
        mt19937 rng(42);uniform_int_distribution<int>ui(0,NR-1);int NQT=NMAT*NQ;
        vector<f32>qc(NQT*N),qt(NQT);
        for(int m=0;m<NMAT;m++)for(int i=0;i<NQ;i++){int idx=ui(rng);
            memcpy(&qc[(m*NQ+i)*N],&ms[m][idx*N],N*4);qt[m*NQ+i]=mt[m][idx];}
        // known KDT-601D values
        f32 kdt_lat_known[]={29,23161,23246};
        f32 kdt_p1_known[]={100.0f,67.5f,60.5f};
        for(int ni=0;ni<3;ni++){
            f32 nlev=ni==0?0:(ni==1?0.005f:0.01f);
            printf("  %s\n",ni==0?"Clean":ni==1?"0.5%":"1.0%");
            vector<f32>qn(NQT*N);mt19937 ng(123);normal_distribution<f32>nd2(0,1);
            for(int i=0;i<NQT*N;i++){f32 v=qc[i]+nlev*nd2(ng);if(v<0)v=0;if(v>1)v=1;qn[i]=v;}
            // BF
            auto t1=chrono::high_resolution_clock::now();f32 bf_p1=0;
            for(int qi=0;qi<NQT;qi++){int m=qi/NQ;f32 bd=1e30f,bt=0;
                for(int i=0;i<NR;i++){f32 d=d2(&qn[qi*N],&ms[m][i*N],N);if(d<bd){bd=d;bt=mt[m][i];}}
                if(fabs(bt-qt[qi])<=1)bf_p1++;}
            double bf_lat=chrono::duration<double,micro>(chrono::high_resolution_clock::now()-t1).count()/NQT;
            printf("    BF:     P1=%.1f%% lat=%.0fus\n",100*bf_p1/NQT,bf_lat);
            printf("    KDT:   P1=%.1f%% lat=%.0fus (known)\n",kdt_p1_known[ni],kdt_lat_known[ni]);
            // Adaptive
            t1=chrono::high_resolution_clock::now();f32 cf_p1=0;int small=0,large=0;
            for(int qi=0;qi<NQT;qi++){
                f32 qp10[DCOARSE];pcproj(&qn[qi*N],qp10,gm.data(),gc10.data(),DCOARSE);
                // PCAR10D reconstruct 601D -> compute RMS reconstruction error
                f32 rms_recon=0;
                for(int j=0;j<601;j++){
                    f32 recon=gm[j];
                    for(int d=0;d<DCOARSE;d++)recon+=qp10[d]*gc10[d*601+j];
                    f32 diff=qn[qi*601+j]-recon;rms_recon+=diff*diff;
                }
                rms_recon=sqrtf(rms_recon/601.0f);
                int ncand=(rms_recon<RECON_THRESH)?NCAND_BASE:NCAND_NOISE;
                vector<size_t>ri(ncand);vector<f32>rd(ncand);
                nanoflann::KNNResultSet<f32>rs(ncand);rs.init(&ri[0],&rd[0]);
                kdt10.findNeighbors(rs,qp10,nanoflann::SearchParameters());
                f32 bd=1e30f,bt=0;
                for(int k=0;k<ncand;k++){int idx=(int)ri[k];f32 d2v=d2(&qn[qi*N],&as[idx*N],N);if(d2v<bd){bd=d2v;bt=at[idx];}}
                if(fabs(bt-qt[qi])<=1)cf_p1++;if(ncand==NCAND_BASE)small++;else large++;
            }
            double cf_lat=chrono::duration<double,micro>(chrono::high_resolution_clock::now()-t1).count()/NQT;
            printf("    Adapt: P1=%.1f%% lat=%.0fus (s=%d l=%d) vsKDT:%.0fx\n",100*cf_p1/NQT,cf_lat,small,large,kdt_lat_known[ni]/(f32)cf_lat);
        }
    }
    printf("Done\n");return 0;
}
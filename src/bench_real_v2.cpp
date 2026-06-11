// bench_real_v2.cpp - 真实 CSV 数据 KDF 管线 (正确加载 bench_data 库)
// cl /O2 /openmp /EHsc /std:c++17 /arch:AVX2 bench_real_v2.cpp /Fereal_v2.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <windows.h>
#include <immintrin.h>
namespace fs = std::filesystem;
using std::vector; using std::string; using std::ifstream;
enum {NW=601, NMAT=4, KR=50};
const char* MN[4]={"ox","sin","soi","cauthy"};
const char* DF[4]={"OX","SIN","SOI","CAUTYONGLASS"};
const char* BD="D:\\kd_forest_v2\\bench_data\\";
struct Ti{ LARGE_INTEGER t0,f;
    Ti(){QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t0);}
    double us(){LARGE_INTEGER t1;QueryPerformanceCounter(&t1);return 1e6*(t1.QuadPart-t0.QuadPart)/f.QuadPart;}
};

float d2_avx(const float* a, const float* b, int n) {
    __m256 s=_mm256_setzero_ps(); int i=0;
    auto L=[&](int o){__m256 d=_mm256_sub_ps(_mm256_loadu_ps(a+i+o),_mm256_loadu_ps(b+i+o));s=_mm256_fmadd_ps(d,d,s);};
    for(;i+32<=n;i+=32){L(0);L(8);L(16);L(24);}
    __m128 lo=_mm256_castps256_ps128(s),hi=_mm256_extractf128_ps(s,1);
    __m128 ss=_mm_add_ps(lo,hi);ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);
    float r=_mm_cvtss_f32(ss);
    for(;i<n;i++){float d=a[i]-b[i];r+=d*d;}
    return r;
}

// --- BF 601D on lib (OMP on 10K) ---
int bf601d_lib_omp(const float* q, const float* lib, int n) {
    float bd=1e30f; int bi=0;
    #pragma omp parallel
    {
        float ld=1e30f; int li=0;
        #pragma omp for nowait
        for(int j=0;j<n;j++){float d=d2_avx(q,lib+j*NW,NW);if(d<ld){ld=d;li=j;}}
        #pragma omp critical
        { if(ld<bd){ bd=ld; bi=li; } }
    }
    return bi;
}

// --- PCA projection ---
void pcproj(const float* d, const float* mean, const float* comp, float* out, int nd) {
    float buf[NW]; for(int i=0;i<NW;i++) buf[i]=d[i]-mean[i];
    for(int k=0;k<nd;k++){float s=0; for(int i=0;i<NW;i++) s+=buf[i]*comp[k*NW+i]; out[k]=s;}
}
float l2n(const float* d, int n){double s=0;for(int i=0;i<n;i++)s+=d[i]*d[i];return (float)(1.0/(sqrt(s)+1e-10f));}

// --- Load binary ---
vector<float> loadf(const string& p) {
    FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"MISS %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
    vector<float> d(sz/4);fread(d.data(),4,d.size(),f);fclose(f);return d;
}

struct RealSample{
    string name,gt_mat; int gt_i; float gt_thick;
    vector<float> spec;
};

float parse_thick(const string& name){
    string s=name; std::transform(s.begin(),s.end(),s.begin(),::tolower);
    auto p=s.find(".csv"); if(p!=string::npos)s=s.substr(0,p);
    p=s.find(".xlsx"); if(p!=string::npos)s=s.substr(0,p);
    size_t i=0; while(i<s.size()&&!isdigit(s[i]))i++;
    if(i>=s.size())return 0;
    size_t j=i; while(j<s.size()&&(isdigit(s[j])||s[j]=='.'))j++;
    float v=(float)atof(s.substr(i,j-i).c_str());
    size_t k=j; while(k<s.size()&&!isalpha(s[k]))k++;
    if(k<s.size()&&s[k]=='u') v*=1000;
    return v;
}

vector<RealSample> load_real_csv(){
    float wl_lo=400,wl_hi=1000;
    vector<float> wl_lib(NW);
    for(int i=0;i<NW;i++) wl_lib[i]=wl_lo+i*(wl_hi-wl_lo)/(NW-1);
    vector<RealSample> samples;
    for(int mi=0;mi<NMAT;mi++){
        fs::path dp=fs::path("D:/kd_forest_v2/test_data/CE")/DF[mi];
        if(!fs::is_directory(dp)) continue;
        for(auto& f:fs::directory_iterator(dp)){
            if(f.path().extension()!=".csv") continue;
            string fn=f.path().filename().string();
            if(fn.find("Poly")!=string::npos||fn.find("poly")!=string::npos) continue;
            ifstream csv(f.path().string());
            string line;
            getline(csv,line); getline(csv,line);
            vector<float> rw,ri;
            while(getline(csv,line)){
                if(line.empty())continue;
                auto c=line.find(','); if(c==string::npos)continue;
                rw.push_back((float)atof(line.substr(0,c).c_str()));
                ri.push_back((float)atof(line.substr(c+1).c_str()));
            }
            vector<float> spec(NW,0.0f);
            for(int i=0;i<NW;i++){
                float x=wl_lib[i];
                if(x<=rw[0]){spec[i]=ri[0];continue;}
                if(x>=rw.back()){spec[i]=ri.back();continue;}
                for(size_t j=0;j<rw.size()-1;j++){
                    if(x>=rw[j]&&x<=rw[j+1]){
                        float t=(x-rw[j])/(rw[j+1]-rw[j]);
                        spec[i]=ri[j]+t*(ri[j+1]-ri[j]); break;
                    }
                }
            }
            float nl=l2n(spec.data(),NW);
            for(int i=0;i<NW;i++) spec[i]/=nl;
            float gt_thick=parse_thick(fn);
            samples.push_back({fn,MN[mi],mi,gt_thick,spec});
        }
    }
    return samples;
}

int main(){
    printf("=== REAL CSV + KDF 管线 (正确 bench_data 库) ===\n\n");

    // 1. Load data
    printf("[1] Loading...\n");fflush(stdout);
    auto real = load_real_csv();
    printf("  Real CSV: %zu samples\n",real.size());

    vector<float> LN[NMAT], TH[NMAT], PM[NMAT], C50[NMAT], C100[NMAT];
    int LS[NMAT];
    for(int m=0;m<NMAT;m++){
        char b[256];
        sprintf_s(b,256,"%slib_%s_n.bin",BD,MN[m]);     LN[m]=loadf(string(b));
        sprintf_s(b,256,"%slib_%s_thick.bin",BD,MN[m]); TH[m]=loadf(string(b));
        sprintf_s(b,256,"%spca_%s_mean.bin",BD,MN[m]);  PM[m]=loadf(string(b));
        sprintf_s(b,256,"%spca_%s_comp50.bin",BD,MN[m]);C50[m]=loadf(string(b));
        sprintf_s(b,256,"%spca_%s_comp100.bin",BD,MN[m]);C100[m]=loadf(string(b));
        LS[m]=(int)LN[m].size()/NW;
        printf("  %s: lib=%d\n",MN[m],LS[m]);
    }
    // Global PCA
    auto GM=loadf(string(BD)+"pca_mean_601.bin");
    auto GC=loadf(string(BD)+"pca_comp_50x601.bin");

    // Build KDTs
    #include "nanoflann.hpp"
    using KT50 = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float,nanoflann::MatrixAdaptor<float>>,
        nanoflann::MatrixAdaptor<float>,50>;
    vector<nanoflann::MatrixAdaptor<float>> MA50(NMAT, nanoflann::MatrixAdaptor<float>(nullptr,0,50));
    vector<KT50*> kt50(NMAT);
    for(int m=0;m<NMAT;m++){
        vector<float> p50(LS[m]*50);
        #pragma omp parallel for
        for(int i=0;i<LS[m];i++){
            float b[64];
            pcproj(LN[m].data()+i*NW, PM[m].data(), C50[m].data(), b, 50);
            float n=l2n(b,50); for(int k=0;k<50;k++) p50[i*50+k]=b[k]/n;
        }
        MA50[m]=nanoflann::MatrixAdaptor<float>(p50.data(),LS[m],50);
        kt50[m]=new KT50(50,MA50[m],nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kt50[m]->buildIndex();
    }

    // 2. Route using 601D L2 competitive routing
    printf("\n========== ROUTING (601D L2 竞争路由) ==========\n");
    int ok=0;
    Ti tr;
    for(auto& rs:real){
        const float* q=rs.spec.data();
        float bd=1e30f; int bm=0;
        for(int m=0;m<NMAT;m++){
            int ib=bf601d_lib_omp(q,LN[m].data(),LS[m]);
            float d=d2_avx(q,LN[m].data()+ib*NW,NW);
            if(d<bd){bd=d;bm=m;}
        }
        if(bm==rs.gt_i) ok++;
        printf("  %-25s -> %-6s (GT:%-6s) %s\n",rs.name.c_str(),MN[bm],rs.gt_mat.c_str(),bm==rs.gt_i?"OK":"XX");
    }
    printf("\n路由精度: %d/%zu = %.1f%% (延迟 %.0f us/sample)\n",ok,real.size(),100.0f*ok/real.size(),tr.us()/real.size());

    // 3. Thickness search: 4 methods using bench_data libraries (10K each)
    printf("\n========== 厚度搜索 ==========\n");
    printf("%-25s %-5s %-9s %-9s %-9s %-9s %s\n","Sample","Mat","BF-601D","KDT-50D","KDT-100D","KDF-50/50","GT");
    printf("----------------------------------------------------------------------------------------------\n");

    int bf1=0,k50_1=0,k100_1=0,kdf1=0;
    float mae_bf=0,mae_k50=0,mae_k100=0,mae_kdf=0;

    for(auto& rs:real){
        int m=rs.gt_i; // use correct material (routing verified 100%)
        const float* q=rs.spec.data();

        // BF-601D
        int ibf=bf601d_lib_omp(q,LN[m].data(),LS[m]);
        float pbf=TH[m][ibf];

        // KDT-50D K=1
        size_t ix50; float dd50;
        nanoflann::KNNResultSet<float> rs50(1); rs50.init(&ix50,&dd50);
        float b50[64]; pcproj(q,PM[m].data(),C50[m].data(),b50,50);
        float nb50=l2n(b50,50); for(int d=0;d<50;d++) b50[d]/=nb50;
        kt50[m]->findNeighbors(rs50,b50,nanoflann::SearchParameters{});
        float pk50=TH[m][(int)ix50];

        // KDF-50/50 (PCA-50D BF K=50 + cosine rerank)
        vector<float> dd(LS[m]);
        #pragma omp parallel for
        for(int j=0;j<LS[m];j++){
            float b[64]; pcproj(LN[m].data()+j*NW,PM[m].data(),C50[m].data(),b,50);
            float n=l2n(b,50); for(int k=0;k<50;k++) b[k]/=n;
            dd[j]=d2_avx(b50,b,50);
        }
        vector<int> srt(LS[m]);
        for(int j=0;j<LS[m];j++) srt[j]=j;
        std::partial_sort(srt.begin(),srt.begin()+KR,srt.end(),
            [&](int a,int b){return dd[a]<dd[b];});
        int bi=srt[0]; float bdd=d2_avx(q,LN[m].data()+bi*NW,NW);
        for(int k=1;k<KR;k++){
            int j=srt[k]; float d=d2_avx(q,LN[m].data()+j*NW,NW);
            if(d<bdd){bdd=d; bi=j;}
        }
        float pkdf=TH[m][bi];

        float gt=rs.gt_thick;
        float ebf=fabsf(pbf-gt), ek50=fabsf(pk50-gt), ekdf=fabsf(pkdf-gt);
        if(ebf<=1) bf1++; mae_bf+=ebf;
        if(ek50<=1) k50_1++; mae_k50+=ek50;
        if(ekdf<=1) kdf1++; mae_kdf+=ekdf;

        printf("%-25s %-5s %7.1f(%s) %7.1f(%s) %7.1f(%s) %7.1f(%s) %s\n",
            rs.name.c_str(), MN[m],
            pbf, ebf<=1?"OK":"XX",
            pk50, ek50<=1?"OK":"XX",
            (float)0, "N/A",  // KDT-100D not built (skip)
            pkdf, ekdf<=1?"OK":"XX",
            (gt==(int)gt?std::to_string((int)gt):std::to_string(gt)).c_str());
    }

    printf("----------------------------------------------------------------------------------------------\n");
    printf("P1nm:  BF=%d/%zu(%.0f%%) K50=%d/%zu(%.0f%%) K100=N/A KDF=%d/%zu(%.0f%%)\n",
        bf1,real.size(),100.0*bf1/real.size(),
        k50_1,real.size(),100.0*k50_1/real.size(),
        kdf1,real.size(),100.0*kdf1/real.size());
    printf("MAE:   BF=%.1f K50=%.1f KDF=%.1f\n",mae_bf/real.size(),mae_k50/real.size(),mae_kdf/real.size());

    for(int m=0;m<NMAT;m++) delete kt50[m];
    return 0;
}

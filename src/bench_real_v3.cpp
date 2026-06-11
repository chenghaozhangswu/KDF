// bench_real_v3.cpp - 真实CSV + bench_data正确库
// cl /O2 /openmp /EHsc /std:c++17 /arch:AVX2 /source-charset:utf-8 bench_real_v3.cpp /Fereal_v3.exe
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
    for(;i+32<=n;i+=32){
        __m256 d0=_mm256_sub_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i));
        __m256 d1=_mm256_sub_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8));
        __m256 d2=_mm256_sub_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16));
        __m256 d3=_mm256_sub_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24));
        s=_mm256_fmadd_ps(d0,d0,s); s=_mm256_fmadd_ps(d1,d1,s);
        s=_mm256_fmadd_ps(d2,d2,s); s=_mm256_fmadd_ps(d3,d3,s);
    }
    __m128 lo=_mm256_castps256_ps128(s),hi=_mm256_extractf128_ps(s,1);
    __m128 ss=_mm_add_ps(lo,hi);ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);
    float r=_mm_cvtss_f32(ss);
    for(;i<n;i++){float d=a[i]-b[i];r+=d*d;}
    return r;
}

int bf_omp(const float* q, const float* lib, int n) {
    float bd=1e30f; int bi=0;
    #pragma omp parallel
    { float ld=1e30f; int li=0;
      #pragma omp for nowait
      for(int j=0;j<n;j++){float d=d2_avx(q,lib+(size_t)j*NW,NW);if(d<ld){ld=d;li=j;}}
      #pragma omp critical
      {if(ld<bd){bd=ld;bi=li;}} }
    return bi;
}

vector<float> loadf(const string& p) {
    FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"MISS %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
    vector<float> d(sz/4);fread(d.data(),4,d.size(),f);fclose(f);return d;
}

float l2n(const float* d, int n){double s=0;for(int i=0;i<n;i++)s+=d[i]*d[i];return (float)(1.0/(sqrt(s)+1e-10f));}
void pcproj(const float* d, const float* m, const float* c, float* o, int nd){
    float b[NW]; for(int i=0;i<NW;i++) b[i]=d[i]-m[i];
    for(int k=0;k<nd;k++){float s=0;for(int i=0;i<NW;i++)s+=b[i]*c[(size_t)k*NW+i];o[k]=s;}
}

float parse_thick(const string& name){
    string s=name; std::transform(s.begin(),s.end(),s.begin(),::tolower);
    auto p=s.find(".csv"); if(p!=string::npos)s=s.substr(0,p);
    size_t i=0; while(i<s.size()&&!isdigit(s[i]))i++;
    if(i>=s.size())return 0;
    size_t j=i; while(j<s.size()&&(isdigit(s[j])||s[j]=='.'))j++;
    float v=(float)atof(s.substr(i,j-i).c_str());
    size_t k=j; while(k<s.size()&&!isalpha(s[k]))k++;
    if(k<s.size()&&s[k]=='u') v*=1000;
    return v;
}

int main(){
    printf("=== REAL CSV + KDF (bench_data correct TMM) ===\n\n");

    printf("[1] Load real CSV...\n");fflush(stdout);
    float wl_lo=400,wl_hi=1000;
    vector<float> wli(NW);
    for(int i=0;i<NW;i++) wli[i]=wl_lo+i*(wl_hi-wl_lo)/(NW-1);
    struct R{string fn,gt; int gi; float gt2; vector<float> s;};
    vector<R> real;
    for(int mi=0;mi<NMAT;mi++){
        fs::path dp=fs::path("D:/kd_forest_v2/test_data/CE")/DF[mi];
        if(!fs::is_directory(dp)) continue;
        for(auto& f:fs::directory_iterator(dp)){
            if(f.path().extension()!=".csv") continue;
            string fn=f.path().filename().string();
            if(fn.find("Poly")!=string::npos||fn.find("poly")!=string::npos) continue;
            ifstream csv(f.path().string()); string line;
            getline(csv,line); getline(csv,line);
            vector<float> rw,ri;
            while(getline(csv,line)){if(line.empty())continue;auto c=line.find(',');if(c==string::npos)continue;rw.push_back((float)atof(line.substr(0,c).c_str()));ri.push_back((float)atof(line.substr(c+1).c_str()));}
            vector<float> spec(NW,0.0f);
            for(int i=0;i<NW;i++){
                float x=wli[i];
                if(x<=rw[0]){spec[i]=ri[0];continue;}
                if(x>=rw.back()){spec[i]=ri.back();continue;}
                for(size_t j=0;j<rw.size()-1;j++){if(x>=rw[j]&&x<=rw[j+1]){float t=(x-rw[j])/(rw[j+1]-rw[j]);spec[i]=ri[j]+t*(ri[j+1]-ri[j]);break;}}
            }
            float nl=l2n(spec.data(),NW); for(int i=0;i<NW;i++) spec[i]/=nl;
            real.push_back({fn,MN[mi],mi,parse_thick(fn),spec});
        }
    }
    printf("  %zu samples\n",real.size());

    printf("[2] Load bench_data libraries...\n");fflush(stdout);
    vector<float> LN[NMAT],TH[NMAT],PM[NMAT],C50[NMAT];
    int LS[NMAT];
    for(int m=0;m<NMAT;m++){
        char b[256];
        sprintf_s(b,256,"%slib_%s_n.bin",BD,MN[m]);     LN[m]=loadf(string(b));
        sprintf_s(b,256,"%slib_%s_thick.bin",BD,MN[m]); TH[m]=loadf(string(b));
        sprintf_s(b,256,"%spca_%s_mean.bin",BD,MN[m]);  PM[m]=loadf(string(b));
        sprintf_s(b,256,"%spca_%s_comp50.bin",BD,MN[m]);C50[m]=loadf(string(b));
        LS[m]=(int)LN[m].size()/NW;
        printf("  %s: lib=%d\n",MN[m],LS[m]);
    }

    printf("[3] Precompute PCA-50D for full lib...\n");fflush(stdout);
    vector<float> P50[NMAT];
    for(int m=0;m<NMAT;m++){
        P50[m].resize(LS[m]*50);
        #pragma omp parallel for
        for(int i=0;i<LS[m];i++){
            float b[64];
            pcproj(LN[m].data()+(size_t)i*NW,PM[m].data(),C50[m].data(),b,50);
            float n=l2n(b,50); for(int k=0;k<50;k++) P50[m][(size_t)i*50+k]=b[k]/n;
        }
    }

    // ===== ROUTING =====
    printf("\n========== ROUTING ==========\n");
    int rok=0; Ti tr;
    for(auto& rs:real){
        int bm=0; float bd=1e30f;
        for(int m=0;m<NMAT;m++){
            int ib=bf_omp(rs.s.data(),LN[m].data(),LS[m]);
            float d=d2_avx(rs.s.data(),LN[m].data()+(size_t)ib*NW,NW);
            if(d<bd){bd=d;bm=m;}
        }
        if(bm==rs.gi) rok++;
        printf("  %-25s -> %-6s (GT:%-6s) %s\n",rs.fn.c_str(),MN[bm],rs.gt.c_str(),bm==rs.gi?"OK":"XX");
    }
    printf("\n  Routing: %d/%zu = %.1f%%, %.0f us/sample\n",rok,real.size(),100.0f*rok/real.size(),tr.us()/real.size());

    // ===== THICKNESS =====
    printf("\n========== THICKNESS ==========\n");
    printf("%-25s %-5s  %-15s %-15s %-15s %s\n","Sample","Mat","BF-601D","KDT-50D","KDF-50/50","GT");
    printf("-------------------------------------------------------------------------\n");

    int bf1=0,k50_1=0,kdf1=0;
    float mbf=0,mk50=0,mkdf=0;
    Ti tt;

    for(auto& rs:real){
        int m=rs.gi;
        const float* q=rs.s.data();

        // BF-601D
        int ibf=bf_omp(q,LN[m].data(),LS[m]);
        float pbf=TH[m][ibf];

        // KDT-50D (PCA-50D BF)
        float bq[64]; pcproj(q,PM[m].data(),C50[m].data(),bq,50);
        float nbq=l2n(bq,50); for(int k=0;k<50;k++) bq[k]/=nbq;
        int ib50=0; float bd50=1e30f;
        for(int i=0;i<LS[m];i++){
            float d=d2_avx(bq,P50[m].data()+(size_t)i*50,50);
            if(d<bd50){bd50=d;ib50=i;}
        }
        float pk50=TH[m][ib50];

        // KDF-50/50
        vector<std::pair<float,int>> top(KR,{1e30f,0});
        for(int i=0;i<LS[m];i++){
            float d=d2_avx(bq,P50[m].data()+(size_t)i*50,50);
            if(d<top.back().first){top.back()={d,i};std::sort(top.begin(),top.end());}
        }
        int bij=top[0].second; float bdd=d2_avx(q,LN[m].data()+(size_t)bij*NW,NW);
        for(int k=1;k<KR;k++){
            int j=top[k].second; float d=d2_avx(q,LN[m].data()+(size_t)j*NW,NW);
            if(d<bdd){bdd=d;bij=j;}
        }
        float pkdf=TH[m][bij];

        float gt=rs.gt2;
        float ebf=fabsf(pbf-gt), ek50=fabsf(pk50-gt), ekdf=fabsf(pkdf-gt);
        if(ebf<=1) bf1++; mbf+=ebf;
        if(ek50<=1) k50_1++; mk50+=ek50;
        if(ekdf<=1) kdf1++; mkdf+=ekdf;

        char bf_s[8], k50_s[8], kdf_s[8];
        sprintf_s(bf_s,"%s",ebf<=1?"OK":"XX");
        sprintf_s(k50_s,"%s",ek50<=1?"OK":"XX");
        sprintf_s(kdf_s,"%s",ekdf<=1?"OK":"XX");
        printf("%-25s %-5s  %4s ~%6.1fnm %4s ~%6.1fnm %4s ~%6.1fnm %5.0fnm\n",
            rs.fn.c_str(),MN[m],bf_s,pbf,k50_s,pk50,kdf_s,pkdf,gt);
    }
    printf("-------------------------------------------------------------------------\n");
    printf("P1nm: BF=%d/%zu(%.0f%%) K50=%d/%zu(%.0f%%) KDF=%d/%zu(%.0f%%)\n",
        bf1,real.size(),100.0*bf1/real.size(),
        k50_1,real.size(),100.0*k50_1/real.size(),
        kdf1,real.size(),100.0*kdf1/real.size());
    printf("MAE:  BF=%.1f K50=%.1f KDF=%.1f\n",mbf/real.size(),mk50/real.size(),mkdf/real.size());
    printf("Latency: %.0f us/sample (thickness only)\n",tt.us()/real.size());
    return 0;
}
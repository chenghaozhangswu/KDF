// bench_real_paper.cpp - 真实 CSV 数据运行 KDF 管线，结果写入论文
// cl /O2 /openmp /EHsc /std:c++17 /arch:AVX2 bench_real_paper.cpp /Fe:real_paper.exe
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
static const char* ND = "D:\\kd_forest_project\\data\\";
enum {NW=601, NMAT=4, nR=10000};
const char* MN[4]={"ox","sin","soi","cauthy"};
const char* DF[4]={"OX","SIN","SOI","CAUTYONGLASS"};
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

float avx_dot(const float* a, const float* b) {
    __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i+32<=NW;i+=32){s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),s);s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16),s);s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24),s);}
    __m128 lo=_mm256_castps256_ps128(s),hi=_mm256_extractf128_ps(s,1);
    __m128 ss=_mm_add_ps(lo,hi);ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);
    float r=_mm_cvtss_f32(ss);
    for(;i<NW;i++) r+=a[i]*b[i]; return r;
}

struct SpecLib {
    vector<float> L[NW]; // we'll store transposed: L[k][i] = spec[i][k]
    int n;
    void build(const float* d, int nn) {
        n=nn;
        for(int k=0;k<NW;k++){L[k].resize(n);for(int i=0;i<n;i++) L[k][i]=d[(size_t)i*NW+k];}
    }
    float dot(int i, const float* q) const {
        __m256 s=_mm256_setzero_ps(); int k=0;
        for(;k+32<=NW;k+=32){
            s=_mm256_fmadd_ps(_mm256_loadu_ps(L[k].data()+i),_mm256_loadu_ps(q+k),s);
            s=_mm256_fmadd_ps(_mm256_loadu_ps(L[k+8].data()+i),_mm256_loadu_ps(q+k+8),s);
            s=_mm256_fmadd_ps(_mm256_loadu_ps(L[k+16].data()+i),_mm256_loadu_ps(q+k+16),s);
            s=_mm256_fmadd_ps(_mm256_loadu_ps(L[k+24].data()+i),_mm256_loadu_ps(q+k+24),s);
        }
        __m128 lo=_mm256_castps256_ps128(s),hi=_mm256_extractf128_ps(s,1);
        __m128 ss=_mm_add_ps(lo,hi);ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);
        float r=_mm_cvtss_f32(ss);
        for(;k<NW;k++) r+=L[k][i]*q[k];
        return r;
    }
};

struct RealSample {
    string name, gt_mat;
    int gt_mat_i;
    float gt_thick;
    vector<float> spec; // 601D L2-normed
};

float parse_thick(const string& name) {
    string s=name; std::transform(s.begin(),s.end(),s.begin(),::tolower);
    // Remove .csv / .xlsx
    auto p=s.find(".csv"); if(p!=string::npos)s=s.substr(0,p);
    p=s.find(".xlsx"); if(p!=string::npos)s=s.substr(0,p);
    // Extract digits
    size_t i=0; while(i<s.size()&&!isdigit(s[i]))i++;
    if(i>=s.size()) return 0;
    size_t j=i; while(j<s.size()&&(isdigit(s[j])||s[j]=='.'))j++;
    float v=(float)atof(s.substr(i,j-i).c_str());
    // Check for um suffix after the number
    size_t k=j; while(k<s.size()&&!isalpha(s[k]))k++;
    if(k<s.size()&&s[k]=='u') v*=1000; // um -> nm
    return v;
}

vector<RealSample> load_real_csv() {
    float wl_lo=400, wl_hi=1000;
    vector<float> wl_lib(NW);
    for(int i=0;i<NW;i++) wl_lib[i]=wl_lo+i*(wl_hi-wl_lo)/(NW-1);
    vector<RealSample> samples;
    for(int mi=0;mi<NMAT;mi++){
        fs::path dp=fs::path("D:/kd_forest_v2/test_data/CE")/DF[mi];
        if(!fs::is_directory(dp)) continue;
        for(auto& f:fs::directory_iterator(dp)){
            if(f.path().extension()!=".csv") continue;
            string fn=f.path().filename().string();
            // Skip POLY in real CSV directory (we don't have poly in our 4-material library)
            if(fn.find("Poly")!=string::npos||fn.find("poly")!=string::npos) continue;
            ifstream csv(f.path().string());
            string line;
            getline(csv,line); getline(csv,line); // skip header
            vector<float> rw, ri;
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
            // L2 normalize
            double sn=0; for(int i=0;i<NW;i++) sn+=spec[i]*spec[i];
            float inv=1.0f/(sqrtf((float)sn)+1e-10f);
            for(int i=0;i<NW;i++) spec[i]*=inv;
            float gt_thick=parse_thick(fn);
            samples.push_back({fn, MN[mi], mi, gt_thick, spec});
        }
    }
    return samples;
}

// Load binary file
vector<float> loadf(const string& p) {
    FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"MISS %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
    vector<float> d(sz/4);fread(d.data(),4,d.size(),f);fclose(f);return d;
}

// Load full library 601D
vector<float> load_lib_601d() {
    size_t n=(size_t)500000*NMAT;
    vector<float> d(n*NW);
    #pragma omp parallel for
    for(int m=0;m<NMAT;m++){
        string sn=string(ND)+"noisy_q_clean_601d.bin";
        char buf[128]; sprintf_s(buf,128,"lib_%s_601d.bin",MN[m]);
        string fp=string("D:\\kd_forest_v2\\bench_data\\")+buf;
        FILE*f=fopen(fp.c_str(),"rb"); if(!f){fprintf(stderr,"MISS %s\n",fp.c_str());exit(1);}
        size_t start=(size_t)m*NW*500000;
        fread(d.data()+start,4,500000*NW,f);
        fclose(f);
    }
    return d;
}

// Thin: thickness for each sample
vector<float> load_thick() {
    size_t n=(size_t)500000*NMAT;
    vector<float> d(n);
    #pragma omp parallel for
    for(int m=0;m<NMAT;m++){
        char buf[128]; 
        string fp;
        // Try bench_data first
        sprintf_s(buf,128,"D:\\kd_forest_v2\\bench_data\\thick_%s.bin",MN[m]);
        FILE*f=fopen(buf,"rb");
        if(!f){
            // Try project data
            sprintf_s(buf,128,"%slib_thick.bin",ND);
            f=fopen(buf,"rb");
            if(!f){fprintf(stderr,"MISS thick\n");exit(1);}
            size_t start=500000*m;
            fread(d.data()+start,4,500000,f); fclose(f);
        } else {
            fread(d.data()+(size_t)m*500000,4,500000,f); fclose(f);
        }
    }
    return d;
}

int main() {
    printf("======================================================\n");
    printf(" REAL CSV 数据 | 全 KDF 管线测试\n");
    printf("======================================================\n\n");

    // 1. Load real CSV
    printf("[1] Load real CSV data... ");fflush(stdout);
    auto real=load_real_csv();
    printf("%zu samples\n",real.size());fflush(stdout);
    for(auto& r:real)printf("  %-25s mat=%s thick=%.0fnm\n",r.name.c_str(),r.gt_mat.c_str(),r.gt_thick);

    // 2. Load 10K routing sub-samples
    printf("\n[2] Load 10K routing sub-samples... ");fflush(stdout);
    vector<float> R[4]; // 10K x 601D each
    vector<float> RLN[4]; // L2 norms
    for(int m=0;m<NMAT;m++){
        char b[128]; sprintf_s(b,128,"D:\\kd_forest_v2\\bench_data\\lib_%s_n.bin",MN[m]);
        R[m]=loadf(b);
        RLN[m].resize(nR);
        for(int i=0;i<nR;i++){
            double s=0; for(int k=0;k<NW;k++){float v=R[m][(size_t)i*NW+k];s+=v*v;}
            RLN[m][i]=1.0f/(sqrtf((float)s)+1e-10f);
        }
    }
    printf("ok\n");fflush(stdout);

    // 3. Load full library for thickness search
    printf("[3] Load full 500K x 4 library for thickness... ");fflush(stdout);
    auto lib=load_lib_601d();
    auto thick=load_thick();
    // Build SpecLib for fast column access
    SpecLib slib[NMAT];
    for(int m=0;m<NMAT;m++){
        slib[m].build(lib.data()+(size_t)m*NW*500000,500000);
    }
    printf("ok\n");fflush(stdout);

    // 4. Route real CSV data using 601D competitive routing
    printf("\n======================================================\n");
    printf(" STEP 1: 601D L2 竞争路由 (10K子采样)\n");
    printf("======================================================\n");
    int route_ok=0;
    vector<int> route_mat(real.size());
    Ti tr;
    for(size_t si=0;si<real.size();si++){
        const float* q=real[si].spec.data();
        float bd=1e30f; int bm=0;
        for(int m=0;m<NMAT;m++){
            const float* rr=R[m].data();
            float* rln=RLN[m].data();
            for(int i=0;i<nR;i++){
                float d=0; for(int k=0;k<NW;k++){float v=rr[(size_t)i*NW+k];d+=(v-q[k])*(v-q[k]);}
                if(d<bd){bd=d;bm=m;}
            }
        }
        route_mat[si]=bm;
        bool ok=(bm==real[si].gt_mat_i);
        if(ok) route_ok++;
        printf("  %-25s -> %-6s (GT:%-6s) %s\n",real[si].name.c_str(),MN[bm],real[si].gt_mat.c_str(),ok?"O":"X");
    }
    printf("\n  路由精度: %d/%zu = %.1f%% (延迟 %.0f us/样本)\n",route_ok,real.size(),100.0f*route_ok/real.size(),tr.us()/real.size());

    // 5. Thickness search: KDF-50/50 (KDT-50D K=50 + cosine rerank)
    printf("\n======================================================\n");
    printf(" STEP 2: KDF-50/50 厚度搜索 (500K库)\n");
    printf("======================================================\n");
    
    // Compute L2 norms for full library
    vector<float> ln(NMAT*500000);
    #pragma omp parallel for
    for(int i=0;i<(int)(500000ULL*NMAT);i++){
        double s=0; for(int k=0;k<NW;k++){float v=lib[(size_t)i*NW+k];s+=v*v;}
        ln[i]=1.0f/(sqrtf((float)s)+1e-10f);
    }

    printf("\n%-25s %-6s %-6s %10s %10s %s\n","Sample","Routed","GT","Pred/nm","GT/nm","Error");
    printf("--------------------------------------------------------------------\n");
    float sum_err=0; int thick_ok=0;
    Ti tk;
    for(size_t si=0;si<real.size();si++){
        int m=route_mat[si];
        const float* q=real[si].spec.data();
        
        // KDT-50D K=50 (already loaded, use slib for dot product)
        // Since we don't have KDT built here, use full BF approach for comparison
        // BF-601D on routed material's 500K library
        int best_i=-1; float bd=1e30f;
        for(int i=0;i<500000;i++){
            float d=d2_avx(q,lib.data()+((size_t)m*500000+(size_t)i)*NW,NW);
            if(d<bd){bd=d;best_i=i;}
        }
        // Cosine rerank top-50 by BF (since KDT not available here, use BF for ground truth)
        // For KDF-50/50 approximation: retrieve top-50 by BF, then rerank by cosine
        int K=50;
        vector<std::pair<float,int>> top(K,{1e30f,0});
        for(int i=0;i<500000;i++){
            float d=d2_avx(q,lib.data()+((size_t)m*500000+(size_t)i)*NW,NW);
            if(d<top.back().first){top.back()={d,i};std::sort(top.begin(),top.end());}
        }
        // Cosine rerank
        float bc=-2.0f; int bi=0;
        for(int ci=0;ci<K;ci++){
            int idx=top[ci].second;
            float cs=avx_dot(q,lib.data()+((size_t)m*500000+idx)*NW)*ln[(size_t)m*500000+idx];
            if(cs>bc){bc=cs;bi=idx;}
        }
        
        float pred=thick[(size_t)m*500000+bi];
        float gt=real[si].gt_thick;
        float err=fabsf(pred-gt);
        sum_err+=err;
        bool ok=(err<=1);
        if(ok) thick_ok++;
        printf("%-25s %-6s %-6s %8.1fnm %8.0fnm %6.1f %s\n",
            real[si].name.substr(0,24).c_str(),MN[m],real[si].gt_mat.c_str(),
            pred,gt,err,ok?"O":"X");
    }
    printf("--------------------------------------------------------------------\n");
    printf(" P1nm: %d/%zu = %.1f%%\n",thick_ok,real.size(),100.0f*thick_ok/real.size());
    printf(" MAE:  %.1f nm\n",sum_err/real.size());
    printf(" 延迟: %.0f us/样本 (厚度搜索, 含BF-500K串联)\n",tk.us()/real.size());

    // 6. Compare with BF-601D oracle (known material)
    printf("\n======================================================\n");
    printf(" BASELINE对比: BF-601D Oracle (已知材料)\n");
    printf("======================================================\n");
    int bf_ok=0;
    for(size_t si=0;si<real.size();si++){
        int m=real[si].gt_mat_i;
        const float* q=real[si].spec.data();
        int best_i=-1; float bd=1e30f;
        for(int i=0;i<500000;i++){
            float d=d2_avx(q,lib.data()+((size_t)m*500000+(size_t)i)*NW,NW);
            if(d<bd){bd=d;best_i=i;}
        }
        float pred=thick[(size_t)m*500000+best_i];
        float gt=real[si].gt_thick;
        if(fabsf(pred-gt)<=1) bf_ok++;
    }
    printf(" BF-601D Oracle P1nm: %d/%zu = %.1f%%\n",bf_ok,real.size(),100.0f*bf_ok/real.size());

    printf("\n======================================================\n");
    printf(" 完成\n");
    printf("======================================================\n");
    return 0;
}

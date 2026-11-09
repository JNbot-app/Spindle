#include "internal.h"
#include "kernels.h"
#include <cmath>
#include <cstring>
#include <cfenv>
#include <iostream>
#include <random>

static bool decode_tiles() {
    using namespace spindle;
    Tensor t; t.padded=384; t.shape={7,384,0,0};
    std::vector<uint8_t> codes(7*192);
    for(size_t i=0;i<codes.size();++i) codes[i]=uint8_t(i*73+17);
    t.data=codes.data(); t.values.resize(7*3);
    for(size_t i=0;i<16;++i) t.book[i]=(float(i)-7.f)*.0317f;
    t.book[0]=-0.f; t.book[1]=1e-38f;
    for(size_t i=0;i<t.values.size();++i) t.values[i]=float(i%5)*.031f;
    for(bool scaled:{false,true}) {
        std::vector<float> expected(6*384),actual(6*384+2,1234.f);
        scalar_kernels().decode_tile(t,1,6,expected.data(),scaled);
        cpu_kernels().decode_tile(t,1,6,actual.data()+1,scaled);
        if(actual.front()!=1234.f || actual.back()!=1234.f ||
           std::memcmp(expected.data(),actual.data()+1,expected.size()*sizeof(float))) return false;
    }
    return true;
}

int main() {
    using namespace spindle;
    try { (void)cpu_kernels(); } catch(const std::invalid_argument &e) { std::cerr<<e.what()<<'\n';return 77; }
    std::mt19937 random(721);
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        std::fesetround(rounding);
        if(!decode_tiles()) {std::cerr<<"Decoded weight tiles differ\n";return 1;}
        for(size_t n:{1,3,4,7,8,127,128,129,1024}) {
            std::vector<float> actual(n),expected(n);
            for(size_t i=0;i<n;++i) actual[i]=(int(random()%255)-127)*.023f;
            actual[0]=0;expected=actual;
            scalar_kernels().quantize(expected.data(),n);
            // Existing NEON A8 specifies ties-to-even, independent of FE rounding.
            if(std::strncmp(cpu_kernels().name,"neon",4) && rounding!=FE_TONEAREST)
                a8(actual.data(),n);
            else if(rounding==FE_TONEAREST) a8(actual.data(),n);
            else continue;
            if(std::memcmp(actual.data(),expected.data(),n*sizeof(float))) {std::cerr<<"A8 differs n="<<n<<" rounding="<<rounding<<'\n';
                for(size_t j=0;j<n;++j)if(actual[j]!=expected[j]){std::cerr<<j<<": "<<actual[j]<<" vs "<<expected[j]<<'\n';break;}return 1;}
        }
        float x[128],ref[128];
        for(size_t i=0;i<128;++i) x[i]=ref[i]=(int(random()%2001)-1000)*.007f;
        // Independent frozen scalar butterfly geometry.
        for(size_t stride=1;stride<128;stride*=2)
            for(size_t i=0;i<128;i+=2*stride)
                for(size_t j=0;j<stride;++j){float a=ref[i+j],b=ref[i+j+stride];ref[i+j]=a+b;ref[i+j+stride]=a-b;}
        const double factor=1./std::sqrt(128.);
        for(float &v:ref)v=float(double(v)*factor);
        fwht(x,128);
        if(std::memcmp(x,ref,sizeof x)){std::cerr<<"FWHT differs\n";return 1;}
    }
    std::fesetround(FE_TONEAREST);
    std::cout<<cpu_kernels().name<<" FWHT/A8 passed\n";
}

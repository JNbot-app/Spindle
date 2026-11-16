#include "internal.h"
#include "kernels.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace spindle;
int main(){
    try{(void)cpu_kernels();}catch(const std::invalid_argument &e){std::cerr<<e.what()<<'\n';return 77;}
    std::fesetround(FE_TONEAREST);std::mt19937 rng(47);
    for(size_t threads:{1,2,4})for(size_t cols:{1,127,128,129,768,4096,32768}) {
        Tensor t;t.kind=3;t.ndim=2;t.bits=4;t.group=128;t.padded=uint32_t((cols+127)/128*128);
        const size_t rows=cols==768?1403:9,groups=t.padded/128,half=t.padded/2;
        t.shape={uint32_t(rows),uint32_t(cols),0,0};
        std::vector<uint8_t> codes(rows*half);for(auto &c:codes)c=uint8_t(rng());t.data=codes.data();
        for(size_t j=0;j<16;++j)t.book[j]=(int(j)-7)*.037f;
        t.values.resize(rows*groups);for(size_t j=0;j<t.values.size();++j)t.values[j]=(j%17)*.031f;
        std::vector<float> x(cols),input(t.padded),actual(rows),expected(rows);
        for(auto &v:x)v=(int(rng()%255)-127)*.031f;
        cq4_prepare_input(x.data(),cols,t.padded,1,input.data(),nullptr);
        float maximum=0;for(float v:t.book)maximum=std::max(maximum,std::abs(v));
        float book_scale=maximum>0?maximum/127.f:1.f;int8_t book[16];
        for(size_t j=0;j<16;++j)book[j]=int8_t(std::clamp(int(std::round(t.book[j]*(1.f/book_scale))),-128,127));
        std::vector<int8_t> quantized(t.padded);std::vector<float> scales(groups*2);
        for(size_t g=0;g<groups;++g)for(size_t part=0;part<2;++part){size_t offset=part*half+g*64;float max_input=0;
            for(size_t k=0;k<64;++k)max_input=std::max(max_input,std::abs(input[offset+k]));
            float scale=max_input>0?max_input*(1.f/127.f):1.f;scales[g*2+part]=scale*book_scale;
            for(size_t k=0;k<64;++k)quantized[offset+k]=int8_t(std::clamp(int(std::nearbyint(input[offset+k]*(1.f/scale))),-128,127));}
        for(size_t r=0;r<rows;++r)for(size_t g=0;g<groups;++g){int32_t ev=0,od=0;
            for(size_t k=0;k<64;++k){uint8_t p=codes[r*half+g*64+k];ev+=int(book[p&15])*quantized[g*64+k];od+=int(book[p>>4])*quantized[half+g*64+k];}
            expected[r]+=(float(ev)*scales[g*2]+float(od)*scales[g*2+1])*t.values[r*groups+g];}
        Math math(std::make_shared<ProjectionWorkers>(threads));math.project(t,x.data(),1,actual.data());
        if(std::memcmp(actual.data(),expected.data(),rows*sizeof(float))){std::cerr<<"Integer projection differs cols="<<cols<<'\n';return 1;}
        // Partial row ranges and guards exercise worker boundaries too.
        actual.assign(rows+2,1234);math.project(t,x.data(),1,actual.data()+1,1,rows-1);
        if(actual.front()!=1234 || actual.back()!=1234 ||
            std::memcmp(actual.data()+1,expected.data()+1,(rows-1)*sizeof(float)))return 1;
    }
    std::cout<<cpu_kernels().name<<" bounded integer projections passed\n";
}

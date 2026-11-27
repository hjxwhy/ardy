// Numeric BVH kernel for retarget_bones_seed.py. Angles are intrinsic Z-Y-X.
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>

using Mat = std::array<double, 9>;
using Vec = std::array<double, 3>;
using Quat = std::array<double, 4>;  // active rotations, xyzw

static Quat multiply(const Quat& a, const Quat& b) {
    const auto [x,y,z,w] = a;
    const auto [u,v,k,r] = b;
    return {w*u+x*r+y*k-z*v, w*v-x*k+y*r+z*u, w*k+x*v-y*u+z*r, w*r-x*u-y*v-z*k};
}

static Quat conjugate(const Quat& q) {
    return {-q[0], -q[1], -q[2], q[3]};
}

static Quat from_matrix(const Mat& m) {
    // Choose a well-conditioned component, including for 180 degree rotations.
    Quat q{};
    const double trace = m[0] + m[4] + m[8];
    if (trace > 0) {
        double s = 2 * std::sqrt(trace + 1);
        q = {(m[7]-m[5])/s, (m[2]-m[6])/s, (m[3]-m[1])/s, s/4};
    } else {
        int i = m[0] >= m[4] ? 0 : 1;
        if (m[8] > m[3*i+i]) i = 2;
        int j = (i+1)%3, k = (i+2)%3;
        double s = 2 * std::sqrt(1 + m[3*i+i] - m[3*j+j] - m[3*k+k]);
        q[i] = s/4;
        q[j] = (m[3*i+j]+m[3*j+i])/s;
        q[k] = (m[3*i+k]+m[3*k+i])/s;
        q[3] = (m[3*k+j]-m[3*j+k])/s;
    }
    return q;
}

static Vec apply(const Mat& r, const double* v) {
    return {r[0]*v[0]+r[1]*v[1]+r[2]*v[2], r[3]*v[0]+r[4]*v[1]+r[5]*v[2],
            r[6]*v[0]+r[7]*v[1]+r[8]*v[2]};
}
static Mat compose(const Mat& a, const Mat& b) {
    Mat out;
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            out[3*i+j] = a[3*i]*b[j] + a[3*i+1]*b[3+j] + a[3*i+2]*b[6+j];
    return out;
}
static Mat zyx(const double* a) {
    constexpr double rad = 0.017453292519943295;
    const double z=a[0]*rad, y=a[1]*rad, x=a[2]*rad;
    const double cz=cos(z), sz=sin(z), cy=cos(y), sy=sin(y), cx=cos(x), sx=sin(x);
    return {cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx,sz*cy,sz*sy*sx+cz*cx,
            sz*sy*cx-cz*sx,-sy,cy*sx,cy*cx};
}

// Returns zero on success and the 1-based failing frame for malformed data.
extern "C" int convert(const char* data, size_t size, int frames, int joints,
                       const int* channels, const int* parent, const int* active,
                       const int* driver, const int* receiver_parent, const double* offsets,
                       double root_scale, float* positions, float* root,
                       const int* mapped, const double* inverse_reference, float* rotations) {
    if (joints<2 || joints>128) return -1;
    const char* cur=data;
    const char* end=data+size;
    Mat world[128];
    Quat reference[24];
    for (int j=0; j<24; ++j) {
        for (int k=0; k<4; ++k) reference[j][k] = inverse_reference[j*4+k];
    }
    for (int t=0; t<frames; ++t) {
        double translation[2][3]{};
        for (int j=0; j<joints; ++j) {
            double values[6]{};
            for (int k=0; k<channels[j]; ++k) {
                while (cur<end && (*cur==' ' || *cur=='\t')) ++cur;
                if (cur>=end || *cur<=' ') return t+1;
                if (active[j]) {
                    auto parsed=std::from_chars(cur,end,values[k]);
                    if (parsed.ec!=std::errc{} || !std::isfinite(values[k])) return t+1;
                    cur=parsed.ptr;
                } else {
                    while (cur<end && *cur>' ') ++cur;
                }
            }
            if (!active[j]) continue;
            const Mat local=zyx(values+channels[j]-3);
            world[j]=parent[j]<0 ? local : compose(world[parent[j]],local);
            if (j<2) for (int k=0; k<3; ++k) translation[j][k]=values[k];
        }
        while (cur<end && (*cur==' ' || *cur=='\r' || *cur=='\t')) ++cur;
        if (cur!=end && *cur!='\n') return t+1;
        if (cur!=end) ++cur;
        const Vec hips=apply(world[0],translation[1]);
        for (int k=0; k<3; ++k) {
            root[3*t+k]=float((translation[0][k]+hips[k])*root_scale);
            positions[t*72+k]=0;
        }
        for (int j=1; j<24; ++j) {
            const auto bone=apply(world[driver[j]],offsets+3*j);
            for (int k=0; k<3; ++k)
                positions[t*72+3*j+k]=float(positions[t*72+3*receiver_parent[j]+k]+bone[k]);
        }
        Quat global[24];
        for (int j=0; j<24; ++j) {
            global[j] = multiply(from_matrix(world[mapped[j]]), reference[j]);
            Quat local = j==0 ? global[j] : multiply(conjugate(global[receiver_parent[j]]), global[j]);
            float* out = rotations+(t*24+j)*4;
            double sign_test = local[3];
            if (t>0) {
                sign_test=0;
                for (int k=0; k<4; ++k) sign_test += local[k]*out[k-96];
            }
            for (int k=0; k<4; ++k) out[k] = float(sign_test<0 ? -local[k] : local[k]);
        }
    }
    while (cur<end && *cur<=' ') ++cur;
    return cur==end ? 0 : frames+1;
}

// Independent FK from the stored float32 quaternions and the target rest offsets.
// Returns the largest Euclidean joint error in metres, or -1 for invalid input.
extern "C" double check_rotation_fk(int frames, const int* parents, const double* rest_offsets,
                                    const float* rotations, const float* positions) {
    double worst_squared = 0;
    for (int t=0; t<frames; ++t) {
        Quat global[24];
        Vec joint[24]{};
        for (int j=0; j<24; ++j) {
            const float* input = rotations + 4*(24*t+j);
            Quat local{input[0],input[1],input[2],input[3]};
            double squared = 0;
            for (double v : local) squared += v*v;
            if (!std::isfinite(squared) || std::abs(squared-1)>2e-6) return -1;
            const double norm = std::sqrt(squared);
            for (double& v : local) v /= norm;
            const int p = parents[j];
            if (p<0) {
                global[j] = local;
            } else {
                global[j] = multiply(global[p],local);
                const double* offset = rest_offsets + 3*j;
                Quat vector{offset[0],offset[1],offset[2],0};
                Quat bone = multiply(multiply(global[p],vector),conjugate(global[p]));
                for (int k=0; k<3; ++k) joint[j][k] = joint[p][k] + bone[k];
            }
            double error_squared = 0;
            for (int k=0; k<3; ++k) {
                double diff=joint[j][k]-positions[(24*t+j)*3+k];
                error_squared += diff*diff;
            }
            if (!std::isfinite(error_squared)) return -1;
            if (error_squared>worst_squared) worst_squared = error_squared;
        }
    }
    return std::sqrt(worst_squared);
}

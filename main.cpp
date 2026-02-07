#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <dirent.h>
#include <omp.h>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#define PI 3.14159265358979323846

template <typename T>
inline T clamp_val(T v, T min_v, T max_v) {
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

struct Vec2 {
    float x, y;
};

struct Color {
    float r, g, b;
};

class ProgressBar {
    int total;
    int current;
    int width;
public:
    ProgressBar(int t) : total(t), current(0), width(60) {}
    void update() {
        current++;
        if (current > total) current = total;
        float progress = (float)current / total;
        int pos = (int)(width * progress);
        cout << "\r[";
        for (int i = 0; i < width; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " << int(progress * 100.0) << " %" << flush;
    }
    void finish() {
        cout << endl;
    }
};

class FloatImage {
public:
    int w, h;
    vector<float> data;

    FloatImage() : w(0), h(0) {}
    FloatImage(int width, int height) : w(width), h(height) {
        data.resize(w * h * 3, 0.0f);
    }

    void copy_from(const FloatImage& other) {
        w = other.w;
        h = other.h;
        data = other.data;
    }

    inline void set_pixel(int x, int y, float r, float g, float b) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        int idx = (y * w + x) * 3;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
    }

    inline Color get_pixel(int x, int y) const {
        x = clamp_val(x, 0, w - 1);
        y = clamp_val(y, 0, h - 1);
        int idx = (y * w + x) * 3;
        return { data[idx], data[idx + 1], data[idx + 2] };
    }

    inline float get_luma(int x, int y) const {
        Color c = get_pixel(x, y);
        return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
    }

    static float sinc(float x) {
        if (x == 0.0f) return 1.0f;
        x *= (float)PI;
        return sin(x) / x;
    }

    static float lanczos_kernel(float x, int a) {
        if (abs(x) >= a) return 0.0f;
        return sinc(x) * sinc(x / a);
    }

    Color sample_lanczos(float x, float y, int a = 3) const {
        int xi = (int)floor(x);
        int yi = (int)floor(y);
        
        float r = 0, g = 0, b = 0;
        float w_sum = 0;

        for (int j = yi - a + 1; j <= yi + a; j++) {
            for (int i = xi - a + 1; i <= xi + a; i++) {
                if (i < 0 || i >= w || j < 0 || j >= h) continue;
                
                float wx = lanczos_kernel(x - i, a);
                float wy = lanczos_kernel(y - j, a);
                float weight = wx * wy;

                Color c = get_pixel(i, j);
                r += c.r * weight;
                g += c.g * weight;
                b += c.b * weight;
                w_sum += weight;
            }
        }

        if (w_sum == 0.0f) return {0,0,0};
        return { r / w_sum, g / w_sum, b / w_sum };
    }
};

class MotionEstimator {
    int block_size;
    int search_range;

public:
    MotionEstimator(int bs = 32, int sr = 12) : block_size(bs), search_range(sr) {}

    vector<Vec2> calculate_dense_flow(const FloatImage& ref, const FloatImage& trg) {
        int cols = (ref.w + block_size - 1) / block_size;
        int rows = (ref.h + block_size - 1) / block_size;
        vector<Vec2> vectors(cols * rows);

        #pragma omp parallel for collapse(2)
        for (int by = 0; by < rows; by++) {
            for (int bx = 0; bx < cols; bx++) {
                int start_x = bx * block_size;
                int start_y = by * block_size;
                
                float min_sad = 1e9f;
                Vec2 best_vec = {0, 0};

                for (int dy = -search_range; dy <= search_range; dy++) {
                    for (int dx = -search_range; dx <= search_range; dx++) {
                        float sad = 0;
                        int count = 0;
                        
                        for (int iy = 0; iy < block_size; iy++) {
                            for (int ix = 0; ix < block_size; ix++) {
                                int rx = start_x + ix;
                                int ry = start_y + iy;
                                if (rx >= ref.w || ry >= ref.h) continue;

                                int tx = rx + dx;
                                int ty = ry + dy;
                                
                                float l1 = ref.get_luma(rx, ry);
                                float l2 = trg.get_luma(tx, ty);
                                sad += abs(l1 - l2);
                                count++;
                            }
                        }

                        if (count > 0) {
                            sad /= count;
                            if (sad < min_sad) {
                                min_sad = sad;
                                best_vec = { (float)dx, (float)dy };
                            }
                        }
                    }
                }
                vectors[by * cols + bx] = best_vec;
            }
        }
        
        return smooth_vectors(vectors, cols, rows);
    }

    vector<Vec2> smooth_vectors(const vector<Vec2>& input, int cols, int rows) {
        vector<Vec2> output = input;
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                float sx = 0, sy = 0;
                int count = 0;
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        int nx = x + i;
                        int ny = y + j;
                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            sx += input[ny * cols + nx].x;
                            sy += input[ny * cols + nx].y;
                            count++;
                        }
                    }
                }
                output[y * cols + x].x = sx / count;
                output[y * cols + x].y = sy / count;
            }
        }
        return output;
    }
};

class RobustStacker {
    FloatImage accumulator;
    FloatImage weight_map;
    FloatImage reference;
    int scale_factor;
    bool light_accum_mode;

    void normalize_brightness(FloatImage& img, const FloatImage& ref) {
        double sum_ref = 0;
        double sum_img = 0;
        long count = 0;

        #pragma omp parallel for reduction(+:sum_ref, sum_img, count)
        for(int i=0; i<ref.w * ref.h; i++) {
            sum_ref += ref.get_luma(i%ref.w, i/ref.w);
            sum_img += img.get_luma(i%img.w, i/img.w);
            count++;
        }

        if(count == 0 || sum_img < 1.0) return;
        float gain = (float)(sum_ref / sum_img);

        #pragma omp parallel for
        for(int i=0; i<img.data.size(); i++) {
            img.data[i] *= gain;
        }
    }

    Vec2 get_interpolated_flow(const vector<Vec2>& flow, int cols, int rows, float x, float y, int bw, int bh) {
        float gx = x / bw;
        float gy = y / bh;
        int ix = (int)floor(gx - 0.5f);
        int iy = (int)floor(gy - 0.5f);
        
        float fx = gx - 0.5f - ix;
        float fy = gy - 0.5f - iy;

        ix = clamp_val(ix, 0, cols - 1);
        iy = clamp_val(iy, 0, rows - 1);
        int ix2 = clamp_val(ix + 1, 0, cols - 1);
        int iy2 = clamp_val(iy + 1, 0, rows - 1);

        Vec2 v00 = flow[iy * cols + ix];
        Vec2 v10 = flow[iy * cols + ix2];
        Vec2 v01 = flow[iy2 * cols + ix];
        Vec2 v11 = flow[iy2 * cols + ix2];

        float vx = (1-fx)*(1-fy)*v00.x + fx*(1-fy)*v10.x + (1-fx)*fy*v01.x + fx*fy*v11.x;
        float vy = (1-fx)*(1-fy)*v00.y + fx*(1-fy)*v10.y + (1-fx)*fy*v01.y + fx*fy*v11.y;
        return {vx, vy};
    }

public:
    RobustStacker(const FloatImage& base_ref, int scale, bool light_mode) 
        : scale_factor(scale), light_accum_mode(light_mode) {
        reference = base_ref;
        int tw = base_ref.w * scale;
        int th = base_ref.h * scale;
        accumulator = FloatImage(tw, th);
        weight_map = FloatImage(tw, th);
    }

    void process_frame(FloatImage& frame, bool is_ref) {
        if (!light_accum_mode) {
            normalize_brightness(frame, reference);
        }

        int bw = 32; 
        int bh = 32;
        vector<Vec2> flow;
        int cols = 0, rows = 0;

        if (!is_ref) {
            MotionEstimator me(bw, 16);
            flow = me.calculate_dense_flow(reference, frame);
            cols = (reference.w + bw - 1) / bw;
            rows = (reference.h + bh - 1) / bh;
        }

        int tw = accumulator.w;
        int th = accumulator.h;
        float s = (float)scale_factor;

        #pragma omp parallel for
        for (int y = 0; y < th; y++) {
            for (int x = 0; x < tw; x++) {
                float src_x = x / s;
                float src_y = y / s;

                if (!is_ref) {
                    Vec2 f = get_interpolated_flow(flow, cols, rows, src_x, src_y, bw, bh);
                    src_x += f.x;
                    src_y += f.y;
                }

                Color c_warped = frame.sample_lanczos(src_x, src_y);
                float weight = 1.0f;

                if (!is_ref) {
                    Color c_ref_loc = reference.sample_lanczos(x/s, y/s);
                    float diff = abs(c_warped.r - c_ref_loc.r) + 
                                 abs(c_warped.g - c_ref_loc.g) + 
                                 abs(c_warped.b - c_ref_loc.b);
                    
                    float threshold = light_accum_mode ? 80.0f : 25.0f;
                    float falloff = light_accum_mode ? 0.01f : 0.5f;

                    weight = 1.0f / (1.0f + diff * diff * falloff);
                    if (diff > threshold) weight = 0.0f;
                } else {
                    weight = light_accum_mode ? 1.0f : 3.0f;
                }

                int idx = (y * tw + x) * 3;
                
                #pragma omp atomic
                accumulator.data[idx] += c_warped.r * weight;
                #pragma omp atomic
                accumulator.data[idx+1] += c_warped.g * weight;
                #pragma omp atomic
                accumulator.data[idx+2] += c_warped.b * weight;
                
                #pragma omp atomic
                weight_map.data[idx] += weight;
                #pragma omp atomic
                weight_map.data[idx+1] += weight;
                #pragma omp atomic
                weight_map.data[idx+2] += weight;
            }
        }
    }

    FloatImage resolve_mid_light() {
        FloatImage result(accumulator.w, accumulator.h);
        vector<float> lumas;
        
        long valid_pixels = 0;
        for (size_t i = 0; i < result.data.size(); i++) {
            if (weight_map.data[i] > 0.0001f) {
                result.data[i] = accumulator.data[i] / weight_map.data[i];
            } else {
                result.data[i] = 0.0f;
            }
        }

        for (int i = 0; i < result.w * result.h; i++) {
            float l = result.get_luma(i % result.w, i / result.w);
            lumas.push_back(l);
            if (l > 0.1f) valid_pixels++;
        }

        float avg_lum = 0;
        if (valid_pixels > 0) {
            double total_luma = 0;
            for(float v : lumas) total_luma += v;
            avg_lum = (float)(total_luma / lumas.size());
        } else {
            avg_lum = 128.0f;
        }

        float target_lum = 110.0f; 
        if (light_accum_mode) target_lum = 140.0f; 
        
        float exposure = target_lum / (avg_lum + 1.0f);

        #pragma omp parallel for
        for (size_t i = 0; i < result.data.size(); i++) {
            float v = result.data[i] * exposure;
            
            v = v / (255.0f + v) * 255.0f;
            
            v = pow(v / 255.0f, 1.0f / 1.1f) * 255.0f; 

            result.data[i] = clamp_val(v, 0.0f, 255.0f);
        }

        FloatImage sharpened(result.w, result.h);
        sharpened.copy_from(result);

        #pragma omp parallel for
        for (int y = 1; y < result.h - 1; y++) {
            for (int x = 1; x < result.w - 1; x++) {
                Color c = result.get_pixel(x, y);
                Color u = result.get_pixel(x, y - 1);
                Color d = result.get_pixel(x, y + 1);
                Color l = result.get_pixel(x - 1, y);
                Color r = result.get_pixel(x + 1, y);
                
                float sharp_factor = 0.5f;

                float val_r = c.r + sharp_factor * (4.0f * c.r - u.r - d.r - l.r - r.r);
                float val_g = c.g + sharp_factor * (4.0f * c.g - u.g - d.g - l.g - r.g);
                float val_b = c.b + sharp_factor * (4.0f * c.b - u.b - d.b - l.b - r.b);

                sharpened.set_pixel(x, y, 
                    clamp_val(val_r, 0.0f, 255.0f), 
                    clamp_val(val_g, 0.0f, 255.0f), 
                    clamp_val(val_b, 0.0f, 255.0f));
            }
        }

        return sharpened;
    }
};

int main() {
    cout << "========================================" << endl;
    cout << "   dataimp ENGINE 2.1.3 \ncopyright (c) 2025-2026 kofe     " << endl;
    cout << "========================================" << endl;
    cout << "Select Mode:" << endl;
    cout << "1. Standard Clarity (Best for day/sharpness)" << endl;
    cout << "2. Light Accumulation (Best for low light/HDR)" << endl;
    cout << "Enter choice 1/2: ";
    
    int choice = 1;
    if (!(cin >> choice)) choice = 1;
    bool light_mode = (choice == 2);

    vector<string> files;
    DIR *dir = opendir(".");
    if (dir) {
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            string n = ent->d_name;
            if (n.length() < 4) continue;
            string ext = n.substr(n.length() - 4);
            for (auto &c : ext) c = tolower(c);
            if ((ext == ".jpg" || ext == ".png") && n.find("SR_") == string::npos) {
                files.push_back(n);
            }
        }
        closedir(dir);
    }
    sort(files.begin(), files.end());

    if (files.empty()) {
        cerr << "Error: No images found (.jpg/.png) in current folder." << endl;
        return 1;
    }

    cout << "Found " << files.size() << " frames." << endl;

    int w, h, c;
    unsigned char* d = stbi_load(files[0].c_str(), &w, &h, &c, 3);
    if (!d) return 1;

    FloatImage ref(w, h);
    for (int i = 0; i < w * h * 3; i++) ref.data[i] = (float)d[i];
    stbi_image_free(d);

    int scale = 2;
    RobustStacker stacker(ref, scale, light_mode);
    stacker.process_frame(ref, true);

    ProgressBar bar(files.size() - 1);
    
    for (size_t i = 1; i < files.size(); i++) {
        int fw, fh, fc;
        unsigned char* data = stbi_load(files[i].c_str(), &fw, &fh, &fc, 3);
        if (data) {
            if (fw == w && fh == h) {
                FloatImage img(fw, fh);
                for (int k = 0; k < fw * fh * 3; k++) img.data[k] = (float)data[k];
                stacker.process_frame(img, false);
            }
            stbi_image_free(data);
        }
        bar.update();
    }
    bar.finish();

    cout << "Processing final high-res image..." << endl;
    FloatImage final_img = stacker.resolve_mid_light();

    vector<unsigned char> out_data(final_img.w * final_img.h * 3);
    for (int i = 0; i < out_data.size(); i++) {
        out_data[i] = (unsigned char)final_img.data[i];
    }

    string mode_tag = light_mode ? "Light" : "Sharp";
    string out_name = "SR_Result_" + mode_tag + "_" + to_string(final_img.w) + "x" + to_string(final_img.h) + ".png";
    stbi_write_png(out_name.c_str(), final_img.w, final_img.h, 3, out_data.data(), final_img.w * 3);

    cout << "Done! Saved as: " << out_name << endl;

    return 0;
}
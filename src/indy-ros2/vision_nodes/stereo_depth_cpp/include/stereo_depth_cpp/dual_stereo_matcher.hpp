#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <algorithm>
#include <vector>

namespace {
inline void stackStereoPair(const cv::Mat& imgA,
                            const cv::Mat& imgB,
                            int gapRows,
                            cv::Mat& stacked)
{
    CV_Assert(imgA.type() == imgB.type());
    CV_Assert(imgA.cols == imgB.cols);

    const int total_rows =
        imgA.rows + gapRows + imgB.rows;

    stacked.create(
        total_rows,
        imgA.cols,
        imgA.type());

    imgA.copyTo(
        stacked.rowRange(0, imgA.rows));

    stacked.rowRange(
        imgA.rows,
        imgA.rows + gapRows).setTo(0);

    imgB.copyTo(
        stacked.rowRange(
            imgA.rows + gapRows,
            total_rows));
}

inline void splitStackedDisparity(
    const cv::Mat& stacked,
    int rowsA,
    int gapRows,
    cv::Mat& dispA,
    cv::Mat& dispB)
{
    dispA = stacked.rowRange(0, rowsA);

    dispB = stacked.rowRange(
        rowsA + gapRows,
        stacked.rows);
}

inline void stackStereoList(
    const std::vector<cv::Mat>& images,
    int gapRows,
    cv::Mat& stacked)
{
    CV_Assert(!images.empty());

    const int type = images.front().type();
    const int cols = images.front().cols;

    int total_rows = 0;

    for (const auto& img : images)
    {
        CV_Assert(img.type() == type);
        CV_Assert(img.cols == cols);

        total_rows += img.rows;
    }

    total_rows += gapRows * (images.size() - 1);

    stacked.create(total_rows, cols, type);

    int row = 0;

    for (size_t i = 0; i < images.size(); ++i)
    {
        const auto& img = images[i];

        img.copyTo(
            stacked.rowRange(row, row + img.rows));

        row += img.rows;

        if (i + 1 < images.size())
        {
            stacked.rowRange(
                row,
                row + gapRows).setTo(0);

            row += gapRows;
        }
    }
}

inline void splitStackedDisparityList(
    const cv::Mat& stacked,
    const std::vector<int>& imageRows,
    int gapRows,
    std::vector<cv::Mat>& disparities)
{
    disparities.resize(imageRows.size());

    int row = 0;

    for (size_t i = 0; i < imageRows.size(); ++i)
    {
        disparities[i] =
            stacked.rowRange(
                row,
                row + imageRows[i]);

        row += imageRows[i] + gapRows;
    }
}
}

struct StereoMaskPair {
    cv::Mat left;
    cv::Mat right;
    double left_coverage = -1.0;
    double right_coverage = -1.0;
};

class DualStereoBM {
public:
    cv::Ptr<cv::StereoBM>      matcher_left;
    cv::Ptr<cv::StereoMatcher> matcher_right;
    int gap_rows = 32;

    void init(int minDisp, int numDisp, int blkSize,
              int preFilterType, int preFilterSize, int preFilterCap,
              int textureThreshold, int uniquenessRatio,
              int speckleWindowSize, int speckleRange, int disp12MaxDiff)
    {
        matcher_left = cv::StereoBM::create(numDisp, blkSize);
        matcher_left->setMinDisparity(minDisp);
        matcher_left->setPreFilterType(preFilterType);
        matcher_left->setPreFilterSize(preFilterSize);
        matcher_left->setPreFilterCap(preFilterCap);
        matcher_left->setTextureThreshold(textureThreshold);
        matcher_left->setUniquenessRatio(uniquenessRatio);
        matcher_left->setSpeckleWindowSize(speckleWindowSize);
        matcher_left->setSpeckleRange(speckleRange);
        matcher_left->setDisp12MaxDiff(disp12MaxDiff);
        matcher_right = cv::ximgproc::createRightMatcher(matcher_left);
        gap_rows = std::max(32, blkSize * 2);
    }

    void compute(
        const cv::Mat& imgLA, const cv::Mat& imgRA,
        const cv::Mat& imgLB, const cv::Mat& imgRB,
        cv::Mat& dispLA, cv::Mat& dispRA,
        cv::Mat& dispLB, cv::Mat& dispRB)
    {
        CV_Assert(imgLA.type() == CV_8UC1 && imgRA.type() == CV_8UC1);
        CV_Assert(imgLB.type() == CV_8UC1 && imgRB.type() == CV_8UC1);
        CV_Assert(imgLA.size() == imgRA.size());
        CV_Assert(imgLB.size() == imgRB.size());
        CV_Assert(imgLA.cols == imgLB.cols);
        CV_Assert(matcher_left && matcher_right);

        cv::Mat left_stacked, right_stacked, disp_left_stacked, disp_right_stacked;
        stackStereoPair(imgLA, imgLB, gap_rows, left_stacked);
        stackStereoPair(imgRA, imgRB, gap_rows, right_stacked);

        matcher_left->compute(left_stacked, right_stacked, disp_left_stacked);
        matcher_right->compute(right_stacked, left_stacked, disp_right_stacked);

        splitStackedDisparity(disp_left_stacked, imgLA.rows, gap_rows, dispLA, dispLB);
        splitStackedDisparity(disp_right_stacked, imgRA.rows, gap_rows, dispRA, dispRB);
    }

    void compute(const std::vector<cv::Mat>& left_images,
                 const std::vector<cv::Mat>& right_images,
                 std::vector<cv::Mat>& left_disparities,
                 std::vector<cv::Mat>& right_disparities)
    {
        CV_Assert(!left_images.empty());
        CV_Assert(left_images.size() == right_images.size());
        CV_Assert(matcher_left && matcher_right);

        std::vector<int> rows;
        rows.reserve(left_images.size());
        for (size_t i = 0; i < left_images.size(); ++i) {
            CV_Assert(left_images[i].type() == CV_8UC1 && right_images[i].type() == CV_8UC1);
            CV_Assert(left_images[i].size() == right_images[i].size());
            CV_Assert(left_images[i].cols == left_images.front().cols);
            rows.push_back(left_images[i].rows);
        }

        cv::Mat left_stacked, right_stacked, disp_left_stacked, disp_right_stacked;
        stackStereoList(left_images, gap_rows, left_stacked);
        stackStereoList(right_images, gap_rows, right_stacked);

        matcher_left->compute(left_stacked, right_stacked, disp_left_stacked);
        matcher_right->compute(right_stacked, left_stacked, disp_right_stacked);

        splitStackedDisparityList(disp_left_stacked, rows, gap_rows, left_disparities);
        splitStackedDisparityList(disp_right_stacked, rows, gap_rows, right_disparities);
    }
};


// ============================================================
//  DualStereoSGBM  —  ghép 2 cặp ảnh theo chiều dọc rồi gọi OpenCV compute 1 lượt
// ============================================================

class DualStereoSGBM {
public:
    cv::Ptr<cv::StereoSGBM>    matcher_left;
    cv::Ptr<cv::StereoMatcher> matcher_right;
    int gap_rows = 32;

    void init(int minDisp, int numDisp, int blkSize,
              int P1, int P2, int disp12MaxDiff,
              int preFilterCap, int uniquenessRatio,
              int speckleWindowSize, int speckleRange, int mode)
    {
        matcher_left = cv::StereoSGBM::create(
            minDisp, numDisp, blkSize, P1, P2,
            disp12MaxDiff, preFilterCap, uniquenessRatio,
            speckleWindowSize, speckleRange, mode);
        matcher_right = cv::ximgproc::createRightMatcher(matcher_left);
        gap_rows = std::max(32, blkSize * 2);
    }

    void compute(
        const cv::Mat& imgLA, const cv::Mat& imgRA,
        const cv::Mat& imgLB, const cv::Mat& imgRB,
        cv::Mat& dispLA, cv::Mat& dispRA,
        cv::Mat& dispLB, cv::Mat& dispRB)
    {
        CV_Assert(imgLA.type() == CV_8UC1 && imgRA.type() == CV_8UC1);
        CV_Assert(imgLB.type() == CV_8UC1 && imgRB.type() == CV_8UC1);
        CV_Assert(imgLA.size() == imgRA.size());
        CV_Assert(imgLB.size() == imgRB.size());
        CV_Assert(imgLA.cols == imgLB.cols);
        CV_Assert(matcher_left && matcher_right);

        cv::Mat left_stacked, right_stacked, disp_left_stacked, disp_right_stacked;
        stackStereoPair(imgLA, imgLB, gap_rows, left_stacked);
        stackStereoPair(imgRA, imgRB, gap_rows, right_stacked);

        matcher_left->compute(left_stacked, right_stacked, disp_left_stacked);
        matcher_right->compute(right_stacked, left_stacked, disp_right_stacked);

        splitStackedDisparity(disp_left_stacked, imgLA.rows, gap_rows, dispLA, dispLB);
        splitStackedDisparity(disp_right_stacked, imgRA.rows, gap_rows, dispRA, dispRB);
    }

    void compute(const std::vector<cv::Mat>& left_images,
                 const std::vector<cv::Mat>& right_images,
                 std::vector<cv::Mat>& left_disparities,
                 std::vector<cv::Mat>& right_disparities)
    {
        CV_Assert(!left_images.empty());
        CV_Assert(left_images.size() == right_images.size());
        CV_Assert(matcher_left && matcher_right);

        std::vector<int> rows;
        rows.reserve(left_images.size());
        for (size_t i = 0; i < left_images.size(); ++i) {
            CV_Assert(left_images[i].type() == CV_8UC1 && right_images[i].type() == CV_8UC1);
            CV_Assert(left_images[i].size() == right_images[i].size());
            CV_Assert(left_images[i].cols == left_images.front().cols);
            rows.push_back(left_images[i].rows);
        }

        cv::Mat left_stacked, right_stacked, disp_left_stacked, disp_right_stacked;
        stackStereoList(left_images, gap_rows, left_stacked);
        stackStereoList(right_images, gap_rows, right_stacked);

        matcher_left->compute(left_stacked, right_stacked, disp_left_stacked);
        matcher_right->compute(right_stacked, left_stacked, disp_right_stacked);

        splitStackedDisparityList(disp_left_stacked, rows, gap_rows, left_disparities);
        splitStackedDisparityList(disp_right_stacked, rows, gap_rows, right_disparities);
    }
};


// ============================================================
//  DualStereoMatcher  —  Wrapper thống nhất
//
//  Thay 4 dòng cũ:
//    left_matcher->compute(obj_L, obj_R, obj_disp_left);
//    right_matcher->compute(obj_R, obj_L, obj_disp_right);
//    left_matcher->compute(obs_L, obs_R, obs_disp_left);
//    right_matcher->compute(obs_R, obs_L, obs_disp_right);
//
//  Bằng 1 dòng:
//    dual_matcher_.compute(obj_L, obj_R, obs_L, obs_R,
//                          obj_disp_left, obj_disp_right,
//                          obs_disp_left, obs_disp_right);
// ============================================================

class DualStereoMatcher {
public:
    enum class Method { BM, SGBM };
    Method method = Method::SGBM;

    DualStereoBM   bm;
    DualStereoSGBM sgbm;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    double wls_lambda = 1000.0;
    double wls_sigma = 0.5;
    double min_mask_coverage = 0.001;

    void init_bm(int numDisp, int blkSize, int minDisp = 0,
                 int preFilterType = cv::StereoBM::PREFILTER_XSOBEL,
                 int preFilterSize = 9, int preFilterCap = 31,
                 int textureThreshold = 10, int uniquenessRatio = 10,
                 int speckleWindowSize = 0, int speckleRange = 2,
                 int disp12MaxDiff = 1)
    {
        method = Method::BM;
        bm.init(minDisp, numDisp, blkSize, preFilterType, preFilterSize,
                preFilterCap, textureThreshold, uniquenessRatio,
                speckleWindowSize, speckleRange, disp12MaxDiff);
        reset_wls_filter();
    }

    void init_sgbm(int minDisp, int numDisp, int blkSize,
                   int P1, int P2, int disp12MaxDiff,
                   int preFilterCap, int uniquenessRatio,
                   int speckleWindowSize, int speckleRange, int mode)
    {
        method = Method::SGBM;
        sgbm.init(minDisp, numDisp, blkSize, P1, P2, disp12MaxDiff,
                  preFilterCap, uniquenessRatio,
                  speckleWindowSize, speckleRange, mode);
        reset_wls_filter();
    }

    void set_wls_params(double lambda, double sigma)
    {
        wls_lambda = lambda;
        wls_sigma = sigma;
        if (wls_filter) {
            wls_filter->setLambda(wls_lambda);
            wls_filter->setSigmaColor(wls_sigma);
        }
    }

    void set_min_mask_coverage(double coverage)
    {
        min_mask_coverage = std::max(0.0, coverage);
    }


    void compute(const std::vector<cv::Mat>& left_images,
                 const std::vector<cv::Mat>& right_images,
                 std::vector<cv::Mat>& left_disparities,
                 std::vector<cv::Mat>& right_disparities)
    {
        if (method == Method::BM)
            bm.compute(left_images, right_images, left_disparities, right_disparities);
        else
            sgbm.compute(left_images, right_images, left_disparities, right_disparities);
    }

    void compute_masked_disparity(const cv::Mat& left_gray,
                                  const cv::Mat& right_gray,
                                  const std::vector<StereoMaskPair>& masks,
                                  double scale,
                                  cv::Mat& merged_disparity)
    {
        CV_Assert(left_gray.type() == CV_8UC1 && right_gray.type() == CV_8UC1);
        CV_Assert(left_gray.size() == right_gray.size());
        CV_Assert(!masks.empty());
        CV_Assert(scale > 0.0);
        CV_Assert(wls_filter);

        merged_disparity.create(left_gray.size(), CV_16S);
        merged_disparity.setTo(0);

        std::vector<cv::Mat> left_clean;
        std::vector<cv::Mat> right_clean;
        std::vector<cv::Mat> left_small;
        std::vector<cv::Mat> right_small;
        std::vector<cv::Mat> active_left_masks;
        left_clean.reserve(masks.size());
        right_clean.reserve(masks.size());
        left_small.reserve(masks.size());
        right_small.reserve(masks.size());
        active_left_masks.reserve(masks.size());

        const double total_pixels = static_cast<double>(left_gray.total());

        for (size_t i = 0; i < masks.size(); ++i) {
            CV_Assert(masks[i].left.type() == CV_8UC1 && masks[i].right.type() == CV_8UC1);
            CV_Assert(masks[i].left.size() == left_gray.size());
            CV_Assert(masks[i].right.size() == right_gray.size());

            const double left_coverage = (masks[i].left_coverage >= 0.0)
                ? masks[i].left_coverage
                : cv::countNonZero(masks[i].left) / total_pixels;
            const double right_coverage = (masks[i].right_coverage >= 0.0)
                ? masks[i].right_coverage
                : cv::countNonZero(masks[i].right) / total_pixels;
            if (left_coverage < min_mask_coverage || right_coverage < min_mask_coverage)
                continue;

            cv::Mat clean_left, clean_right, small_left, small_right;
            clean_left = cv::Mat::zeros(
                left_gray.size(),
                left_gray.type());

            left_gray.copyTo(
                clean_left,
                masks[i].left);
            clean_right = cv::Mat::zeros(
                right_gray.size(),
                right_gray.type());
            right_gray.copyTo(
                clean_right,
                masks[i].right);
            cv::resize(clean_left, small_left, cv::Size(), scale, scale, cv::INTER_NEAREST);
            cv::resize(clean_right, small_right, cv::Size(), scale, scale, cv::INTER_NEAREST);

            left_clean.push_back(std::move(clean_left));
            right_clean.push_back(std::move(clean_right));
            left_small.push_back(std::move(small_left));
            right_small.push_back(std::move(small_right));
            active_left_masks.push_back(masks[i].left);
        }

        if (left_small.empty())
            return;

        std::vector<cv::Mat> disp_left;
        std::vector<cv::Mat> disp_right;

        disp_left.reserve(left_small.size());
        disp_right.reserve(left_small.size());
        compute(left_small, right_small, disp_left, disp_right);

        //cv::Mat filtered;
        for (size_t i = 0; i < left_clean.size(); ++i) {
            wls_filter->filter(disp_left[i], left_clean[i], filtered, disp_right[i]);
            if (filtered.size() != left_gray.size())
                cv::resize(filtered, filtered, left_gray.size(), 0.0, 0.0, cv::INTER_NEAREST);
            filtered.copyTo(merged_disparity, active_left_masks[i]);
        }
    }

    cv::Ptr<cv::StereoMatcher> left_matcher_a() const
    {
        return (method == Method::BM)
            ? bm.matcher_left.dynamicCast<cv::StereoMatcher>()
            : sgbm.matcher_left.dynamicCast<cv::StereoMatcher>();
    }

    cv::Ptr<cv::StereoMatcher> left_matcher_b() const
    {
        return left_matcher_a();
    }

private:
    cv::Mat filtered;
    cv::Ptr<cv::CLAHE> clahe_;

    void preprocessForBM(const cv::Mat& src, cv::Mat& dst,
                          double clahe_clip = 2.0,
                          cv::Size clahe_tile = cv::Size(8, 8),
                          int blur_ksize = 3)
    {
        CV_Assert(src.type() == CV_8UC1);

        // ✅ CLAHE — chỉ trên vùng có mask (pixel > 0)
        // Tránh CLAHE kéo noise vùng đen lên
        if (!clahe_) {
            clahe_ = cv::createCLAHE(clahe_clip, clahe_tile);
        }
        clahe_->apply(src, dst);

        // ✅ Gaussian blur nhẹ — giảm noise cao tần, SAD nhạy cảm với noise
        if (blur_ksize > 1)
            cv::GaussianBlur(dst, dst, cv::Size(blur_ksize, blur_ksize), 0);

        // ✅ Giữ nguyên vùng 0 (background) — không để blur lan vào
        // Vùng background = 0 trong src → mask lại
        dst.setTo(0, src == 0);
    }
    void reset_wls_filter()
    {
        wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher_a());
        set_wls_params(wls_lambda, wls_sigma);
    }
};
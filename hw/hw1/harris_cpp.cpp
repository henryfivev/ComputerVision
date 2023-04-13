#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat src = imread("lena.jpg");
    if (src.empty())
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("input", src);

    Mat gray, dx, dy;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Sobel(gray, dx, CV_32F, 1, 0);
    Sobel(gray, dy, CV_32F, 0, 1);

    int w = src.cols;
    int h = src.rows;
    Mat R = Mat::zeros(src.size(), CV_32FC1);
    float k = 0.04;

    for (int row = 0; row < h; row++)
    {
        for (int col = 0; col < w; col++)
        {
            float dxVal = dx.at<float>(row, col);
            float dyVal = dy.at<float>(row, col);

            float m11 = dxVal * dxVal;
            float m12 = dxVal * dyVal;
            float m22 = dyVal * dyVal;

            Matx22f A(m11, m12, m12, m22);
            float detA = determinant(A);
            float traceA = trace(A);

            float r = detA - k * traceA * traceA;

            R.at<float>(row, col) = r;
        }
    }

    double minVal = 0;
    double maxVal = 0;
    minMaxLoc(R, &minVal, &maxVal);

    Mat dst = src.clone();
    for (int row = 0; row < h; row++)
    {
        for (int col = 0; col < w; col++)
        {
            if (R.at<float>(row, col) > maxVal * 0.01)
            {
                circle(dst, Point(col, row), 5, Scalar(0, 255, 255), 2);
            }
        }
    }

    imshow("harris corner detection", dst);

    waitKey(0);
    return 0;
}
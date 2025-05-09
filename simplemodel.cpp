#include <dlib/svm.h> //err
#include <iostream>

int main() {
    using namespace dlib;
    typedef matrix<double, 1, 1> sample_type;

    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Sample data: y = 2*x + 1
    for (int i = 0; i < 10; ++i) {
        sample_type samp;
        samp(0) = i;
        samples.push_back(samp);
        labels.push_back(2 * i + 1);
    }

    decision_function<linear_kernel<sample_type>> df = train_regression_function(
        svr_trainer<linear_kernel<sample_type>>(), samples, labels
    );

    // Predict
    sample_type test_sample;
    test_sample(0) = 5.5;
    std::cout << "Prediction: " << df(test_sample) << std::endl;

    return 0;
}

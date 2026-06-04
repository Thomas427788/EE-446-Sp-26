#include <TensorFlowLite.h>
#include "network_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define NUMBER_OF_INPUTS 38
#define NUMBER_OF_OUTPUTS 2
#define TENSOR_ARENA_SIZE (128 * 1024)

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::ErrorReporter* error_reporter;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
bool model_ready = false;

#define NUM_TEST_SAMPLES 10

const float X_test[NUM_TEST_SAMPLES][NUMBER_OF_INPUTS] = {
    {
        0.106682595f, 2.21931185f, 0.781427526f, 0.751111287f,
        -0.00773736981f, -0.00489253312f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, -0.717045492f, -0.35434285f,
        -0.637209268f, -0.631929033f, -0.37436224f, -0.374431603f,
        0.771283106f, -0.349683031f, -0.374559704f, 0.734342561f,
        -1.0266544f, -1.13875587f, 2.842716f, 2.43246312f,
        -0.2891034f, -0.639531905f, -0.6248708f, -0.387634624f,
        -0.376387026f, 0.652822878f
    },
    {
        -0.110249223f, -0.124706157f, -0.442083085f, 0.751111287f,
        -0.0077031297f, -0.00479853257f, -0.089486422f, -0.0950756715f,
        1.23569403f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, -0.498720426f, -0.0101587758f,
        -0.637209268f, -0.631929033f, -0.37436224f, -0.374431603f,
        0.771283106f, -0.349683031f, -0.374559704f, -0.354303254f,
        1.25875427f, 1.06640135f, -0.439078168f, -0.447833959f,
        -0.111425688f, -0.639531905f, -0.6248708f, -0.387634624f,
        -0.376387026f, 0.652822878f
    },
    {
        -0.110249223f, 2.21931185f, 0.781427526f, 0.751111287f,
        -0.00776207039f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, -0.586050452f, -0.368110213f,
        -0.637209268f, -0.631929033f, -0.37436224f, -0.374431603f,
        -1.36692173f, 1.25862523f, -0.374559704f, 0.734342561f,
        -1.03568764f, -1.16103019f, 0.937158095f, 2.01174557f,
        -0.2891034f, -0.639531905f, -0.6248708f, 0.264774211f,
        -0.376387026f, 0.216426331f
    },
    {
        -0.110249223f, -0.124706157f, -0.319732024f, -0.73623464f,
        -0.00776224074f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, -0.655914474f, -0.326808124f,
        1.60266389f, 1.60510372f, -0.37436224f, -0.374431603f,
        -0.502541053f, 1.48046085f, -0.374559704f, 0.734342561f,
        -1.00858793f, -1.11648156f, -0.0156208563f, -0.480196848f,
        -0.2891034f, 1.60875908f, 1.6189552f, -0.387634624f,
        -0.376387026f, -0.656366762f
    },
    {
        -0.110249223f, -0.124706157f, -0.870311799f, -0.73623464f,
        -0.00776224074f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, 0.164987774f, -0.134065042f,
        1.60266389f, 1.60510372f, -0.37436224f, -0.374431603f,
        -1.11670627f, -0.0169295977f, -0.374559704f, 0.734342561f,
        -0.891155864f, -1.00510998f, -0.121485184f, -0.480196848f,
        -0.2891034f, 1.60875908f, 1.6189552f, -0.387634624f,
        -0.376387026f, 0.216426331f
    },
    {
        -0.110249223f, -0.124706157f, 0.781427526f, -2.22358057f,
        -0.00776224074f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, 3.72805285f, -0.368110213f,
        -0.480418147f, -0.631929033f, 2.52794924f, 2.71536458f,
        -1.50340289f, 4.9743719f, -0.374559704f, 0.734342561f,
        -1.03568764f, -1.16103019f, 4.37774875f, -0.480196848f,
        -0.2891034f, -0.549600266f, -0.6248708f, 2.74392778f,
        2.75391372f, -0.656366762f
    },
    {
        -0.110249223f, 2.21931185f, 1.08730518f, 0.751111287f,
        -0.00775747097f, -0.00491864438f, 11.7434804f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, -0.00967227892f, 0.747046188f,
        -0.614810536f, -0.631929033f, -0.37436224f, -0.374431603f,
        0.748536246f, -0.23876522f, -0.374559704f, 0.734342561f,
        -0.303995532f, -0.448252096f, -0.386146004f, 0.555415586f,
        -0.2891034f, -0.639531905f, -0.6248708f, -0.387634624f,
        -0.376387026f, -1.09276331f
    },
    {
        -0.108713423f, -0.124706157f, 1.3320073f, 0.751111287f,
        -0.00776121865f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, 0.444443858f, -0.368110213f,
        -0.592411805f, -0.631929033f, 2.43432629f, -0.374431603f,
        -1.48065603f, 5.19620752f, -0.374559704f, 0.734342561f,
        -1.03568764f, -1.16103019f, 2.41925869f, -0.480196848f,
        -0.2891034f, -0.617048995f, -0.6248708f, 1.17814658f,
        -0.376387026f, -1.9655564f
    },
    {
        -0.110249223f, -0.124706157f, -0.686785207f, 0.751111287f,
        -0.00772050528f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        1.23569403f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        0.0285992604f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, -0.717045492f, -0.35434285f,
        -0.637209268f, -0.631929033f, -0.37436224f, -0.374431603f,
        0.771283106f, -0.349683031f, -0.374559704f, -1.71511052f,
        -1.0266544f, -0.782366826f, 0.884225931f, 0.0699722578f,
        -0.2891034f, -0.639531905f, -0.6248708f, -0.387634624f,
        -0.376387026f, 0.216426331f
    },
    {
        -0.110249223f, -0.124706157f, 2.31081579f, -2.22358057f,
        -0.00776224074f, -0.00491864438f, -0.089486422f, -0.0950756715f,
        -0.809261819f, -0.0116636426f, -0.0366518691f, -0.0244365073f,
        -0.0123851504f, -0.0261800242f, -0.0186098963f, -0.0412211976f,
        -0.00281749392f, -0.097530944f, 1.79805927f, -0.161599768f,
        -0.637209268f, -0.631929033f, 2.7464028f, 2.71536458f,
        -1.36692173f, -0.0169295977f, -0.374559704f, 0.734342561f,
        -0.9001891f, -1.02738429f, -0.0685530203f, -0.480196848f,
        -0.2891034f, -0.639531905f, -0.6248708f, 2.87440955f,
        2.75391372f, 0.216426331f
    }
};

const uint8_t y_test[NUM_TEST_SAMPLES] = {1, 1, 0, 0, 0, 0, 0, 0, 1, 0}; // Actual labels for each sample

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    Serial.println("Starting HW2 network anomaly inference sketch...");
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(network_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema version.");
        model_ready = false;
        return;
    }

    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors! Increase TENSOR_ARENA_SIZE or check supported ops.");
        model_ready = false;
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Model initialized successfully.");
    Serial.print("Input type: "); Serial.println(input->type);
    Serial.print("Output type: "); Serial.println(output->type);
    Serial.print("Input bytes: "); Serial.println(input->bytes);
    Serial.print("Output bytes: "); Serial.println(output->bytes);
    model_ready = true;
}

void loop() {
    if (!model_ready) {
        Serial.println("Model is not initialized, so inference is skipped.");
        delay(3000);
        return;
    }

    for (uint8_t i = 0; i < NUM_TEST_SAMPLES; i++) {
        // Load the i-th test sample data into the input tensor.
        // Dynamic-range quantized TFLite models usually still use float32 inputs,
        // but this also handles int8/uint8 inputs just in case.
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
            float value = X_test[i][j];

            if (input->type == kTfLiteFloat32) {
                input->data.f[j] = value;
            } else if (input->type == kTfLiteInt8) {
                int32_t quantized_value = round(value / input->params.scale) + input->params.zero_point;
                quantized_value = max(-128, min(127, quantized_value));
                input->data.int8[j] = (int8_t)quantized_value;
            } else if (input->type == kTfLiteUInt8) {
                int32_t quantized_value = round(value / input->params.scale) + input->params.zero_point;
                quantized_value = max(0, min(255, quantized_value));
                input->data.uint8[j] = (uint8_t)quantized_value;
            }
        }

        // Run the model on this input and check for error.
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Failed to invoke!");
            continue;
        }

        // Problem 5(a): Obtain the prediction from the output tensor and
        // determine the predicted class label. The model has two output scores:
        // class 0 = attack, class 1 = normal.
        float output_scores[NUMBER_OF_OUTPUTS];

        for (int k = 0; k < NUMBER_OF_OUTPUTS; k++) {
            if (output->type == kTfLiteFloat32) {
                output_scores[k] = output->data.f[k];
            } else if (output->type == kTfLiteInt8) {
                output_scores[k] = (output->data.int8[k] - output->params.zero_point) * output->params.scale;
            } else if (output->type == kTfLiteUInt8) {
                output_scores[k] = (output->data.uint8[k] - output->params.zero_point) * output->params.scale;
            } else {
                output_scores[k] = 0.0;
            }
        }

        int predicted_class = 0;
        float prediction = output_scores[0];

        for (int k = 1; k < NUMBER_OF_OUTPUTS; k++) {
            if (output_scores[k] > prediction) {
                prediction = output_scores[k];
                predicted_class = k;
            }
        }

        // Problem 5(b): Output Sample #, Predicted Class, and Actual Class
        // using Serial.print.
        Serial.print("Sample #");
        Serial.print(i + 1);
        Serial.print(" | Predicted Class: ");
        Serial.print(predicted_class);
        Serial.print(" | Actual Class: ");
        Serial.print(y_test[i]);
        Serial.print(" | Output scores: [");
        Serial.print(output_scores[0], 6);
        Serial.print(", ");
        Serial.print(output_scores[1], 6);
        Serial.println("]");

        // Delay between predictions.
        delay(1000);
    }

    // Delay before repeating the tests.
    delay(10000);
}

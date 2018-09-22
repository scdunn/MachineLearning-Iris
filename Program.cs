using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Transforms.Conversions;
using System;

namespace IrisML
{
    public class IrisData
    {
        public float SepalLength;
        public float SepalWidth;
        public float PetalLength;
        public float PetalWidth;
        public string Label;
    }

    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }

    class Program
    {
        const string FILE_DATA_PATH = "irisdata.txt";

        static void Main(string[] args)
        {
            var context = new MLContext();
            
            var reader = context.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.Text, 4)
                }
            });


            IDataView trainingDataView = reader.Read(new MultiFileSource(FILE_DATA_PATH));

          var pipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            
            var model = pipeline.Fit(trainingDataView);

            
            var prediction = model.MakePredictionFunction<IrisData, IrisPrediction>(context).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadKey();
        }
    }
}
package classifier;

import java.awt.BorderLayout;
import java.io.File;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.SMO;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

/**
 *
 * @author Hector E. Dominguez B.
 */
public class Classifier {
    private static int NUM_FOLDS = 5;
    private static double BETA   = 0.5;
    
    // Processed tweets
    private static final String DIR = "../News/Processed";
    private static final String OUT = "src/resources/pro_SVM_input.arff";
    // Raw tweets
    //private static final String DIR = "../News/Raw";
    //private static final String OUT = "src/resources/raw_SVM_input.arff";
    
    private static Instances mInstances, mStructure;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile( new File(OUT) );
            mInstances = loader.getDataSet();
        } catch (NullPointerException ex) { // File not found
            if ( mInstances == null ) {
                System.out.println("Reading Tweets from " + DIR + " directory...");
                fromTweetsToArff(DIR, OUT);
            }
        }
        
        mInstances.setClassIndex(0);
        
        Instances data = getTrainingSet(1000, 1000);
        doCrossValidation(1, data);
    }
    
    private static void fromTweetsToArff(String fromDir, String outputFile) 
            throws Exception {
        
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory( new File(fromDir) );
        Instances rawData = loader.getDataSet();
        //System.out.println("\n\nRaw data:\n\n" + rawData);
                
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(rawData);
        //filter.setOutputWordCounts(true);
        //filter.setWordsToKeep(1000000);
        //filter.setDoNotOperateOnPerClassBasis(true);
        mInstances = Filter.useFilter(rawData, filter);
        mInstances.setRelationName("news");
        //System.out.println("\n\nFiltered data:\n\n" + data);
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(mInstances);
        saver.setFile( new File(outputFile) );
        saver.writeBatch();
    }
    
    private static Instances getTrainingSet(int numInstances, int numAttributes) throws Exception {
        System.out.println("Getting Training Set:");
        
        String[] options;
        Instances newData;
        
        // Get a sample of all instances
        options = new String[] {"-M", "1.0",
                                "-X", Integer.toString(numInstances),
                                "-S", "1" };
        
        SpreadSubsample subSampleFilter = new SpreadSubsample();
        subSampleFilter.setOptions(options);
        subSampleFilter.setInputFormat(mInstances);
        
        System.out.println("\tGetting sample...");
        
        newData = Filter.useFilter(mInstances, subSampleFilter);
        
        // Get top 'numAttributes' attributes
        options = new String[] {"-E", 
                                "weka.attributeSelection.InfoGainAttributeEval ",
                                "-S", 
                                "weka.attributeSelection.Ranker " + 
                                "-T -1.7976931348623157E308 -N "  + 
                                Integer.toString(numAttributes)};
        
        AttributeSelection attributeFilter = new AttributeSelection();
        attributeFilter.setOptions(options);
        attributeFilter.setInputFormat(newData);
        
        System.out.println("\tFiltering attributes...");
        
        newData = Filter.useFilter(newData, attributeFilter);
        
        return newData;
    }
    
    private static void doCrossValidation(int seed, Instances trainingSet) 
            throws Exception {
        System.out.println();
        System.out.println("Doing Cross-Validation...");
        // Randomize data
        Random rand = new Random(seed);
        Instances randData = new Instances(trainingSet);
        randData.randomize(rand);
        if ( randData.classAttribute().isNominal() ) {
            randData.stratify(NUM_FOLDS);
        }
        
        /*
        options = new String[] {"-C", "1.0",
                                "-L", "0.001",
                                "-P", "1.0E-12",
                                "-N", "0",
                                "-V", "-1",
                                "-W","1",
                                "-K", "weka.classifiers.functions.supportVector.PolyKernel " +
                                      "-C 250007 -E 1.0"} ;
                */
        //svm.setOptions(options);
        //svm.buildClassifier(newData);
        
        // Perform cross-validation
        SMO svm, best_svm;
        Instances train, test;
        Evaluation eval, evalAll;
        double best_Fbeta; // Weighted Harmonic Mean
        
        best_svm   = null;
        best_Fbeta = 0.0;
        evalAll    = new Evaluation(randData);
        
        for ( int i = 0; i < NUM_FOLDS; i++ ) {
            eval  = new Evaluation(randData);
            train = randData.trainCV(NUM_FOLDS, i);
            test  = randData.testCV(NUM_FOLDS, i);
            
            // Build and evaluate classifier
            
            svm = new SMO(); // Construct SVM
            svm.buildClassifier(train);
            eval.evaluateModel(svm, test);
            evalAll.evaluateModel(svm, test);
            
            // Output evaluation
            System.out.println();
            System.out.println("Weighted Harmonic Mean: " + eval.weightedFMeasure() );
            System.out.println(eval.toMatrixString("=== Confusion matrix for fold " + 
                                                   (i+1) + "/" + NUM_FOLDS + " ===\n"));
            
            // Get best SVM based on Weighted Harmonic Mean
            if ( best_Fbeta < eval.weightedFMeasure() ) {
                best_Fbeta = eval.weightedFMeasure();
                best_svm = (SMO) SMO.makeCopy(svm);
            }
        }
        
        // Output summary
        System.out.println();
        System.out.println("=== Summary " + NUM_FOLDS + "-fold Cross-validation ===");
        System.out.println(evalAll.toSummaryString());
        System.out.println(evalAll.toClassDetailsString());
        System.out.println(evalAll.toMatrixString());
        
        // Best SVM
        System.out.println();
        System.out.println("=== Best SVM ===");
        System.out.println("Weighted Harmonic Mean: " + best_Fbeta );
        
        /*
        if ( best_svm != null ) {
            System.out.println();
            System.out.println(best_svm.toString());
        }
        */
        //showRocCurve(eval.predictions());
    }

    private static void showRocCurve(FastVector predictions) {
        // Generate Curve
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(predictions, classIndex);
        
        // Plot Curve
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area under ROC = "                           +
                         Utils.doubleToString(tc.getROCArea(result), 4) +
                         ")");
        vmc.setName(result.relationName());
        
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        
        // Specify which points are connected
        boolean[] cp = new boolean[result.numInstances()];
        for (int n = 1; n < cp.length; n++) {
            cp[n] = true;
        }
        
        try {
            tempd.setConnectPoints(cp);
            vmc.addPlot(tempd);
        } catch (Exception ex) {
            System.err.println("Showing ROC Curve...");
            System.exit(-1);
        }
        
        // Display Curve
        String plotName = vmc.getName();
        final javax.swing.JFrame jf;
        
        jf = new javax.swing.JFrame("Weka Classifier Visualize: " + plotName);
        
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });
        
        jf.setVisible(true);
    }
}

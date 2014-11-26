package classifier;

import java.awt.BorderLayout;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SparseInstance;
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
public class CLANews {
    // SVM Parameters
    private static final int BEST_NUM_INSTANCES  = 500;
    private static final int BEST_NUM_ATTRIBUTES = 500;
    private static final int BEST_SEED           = 2000;
    
    private static final int    NUM_FOLDS = 10;
    private static final double SVM_COST  = 1.0;
    private static final double SVM_GAMMA = 0.25;
    
    // Processed tweets
    private static final String DIR       = "../News/Processed";
    private static final String ARFF_FILE = "src/resources/all_tweets.arff";
    private static final int    NUM_CLASS = 5;
    
    // Tuning parameters
    private static final int [] numInstances_list  = new int[] {1000, // per category 
                                                                2000, 
                                                                3000, 
                                                                4000};
    
    private static final int [] numAttributes_list = new int[] {500,
                                                                1000, 
                                                                1500, 
                                                                2000, 
                                                                2500};
    
    // Location trained machine
    private static final String MODEL   = "src/resources/svm.model";
    private static final String DATASET = "src/resources/svm_model_dataset.arff";
    
    // Members variables
    private static Classifier mSVM;
    private static Instances mInstances;
    private static Instances mSVM_dataset;
    
    //private static final String OUT_PLOT = "src/resources/plot/ROC_SVM";
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        ArffLoader loader = new ArffLoader();
        
        // Tuning data parameters
        //getBestDataParameters();
                
        //mSVM = getModel(); // try get model
        
        if ( mSVM == null) {
            // Load all tweets
            try {
                loader.setFile( new File(ARFF_FILE) );
                mInstances = loader.getDataSet();
                System.out.println("Loading ARFF File...");
            } catch (NullPointerException ex) { // File not found
                if ( mInstances == null ) {
                    fromTweetsToArff(DIR, ARFF_FILE);
                }
            }

            mInstances.setClassIndex(0);
            
            // Get sample of all tweets
            mSVM_dataset = getTrainingSet(BEST_NUM_INSTANCES, 
                                               BEST_NUM_ATTRIBUTES);

            mSVM_dataset.setRelationName("svm_model_dataset");
            // Train and Save SVM
            mSVM = doCrossValidation(BEST_SEED, mSVM_dataset);
            saveModel( mSVM );
            // Save dataset used to train SVM
            ArffSaver saver = new ArffSaver();
            saver.setInstances(mSVM_dataset);
            saver.setFile( new File(DATASET) );
            saver.writeBatch();
        } else {
            // Load dataset used to train SVM
            try {
                loader.setFile( new File(DATASET) );
                mSVM_dataset = loader.getDataSet();
            } catch (Exception ex) { // File not found
                System.err.println("Loading ARFF File: " + DATASET);
                System.exit(-1);            
            }

            mSVM_dataset.setClassIndex(mSVM_dataset.numAttributes()-1);
        }
        
        test("../News/Test/tweets.txt");
    }
    
    private static void fromTweetsToArff(String fromDir, String outputFile) 
            throws Exception {
        
        System.out.println("Reading Tweets from directory:" + fromDir );
        
        System.out.println("\tLoading...");
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory( new File(fromDir) );
        Instances rawData = loader.getDataSet();
        
        System.out.println("\tFiltering...");
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(rawData);
        filter.setOutputWordCounts(true); // Word Frequency
        mInstances = Filter.useFilter(rawData, filter);
        mInstances.setRelationName("news");
        
        System.out.println("\tSaving...");
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
    
    private static Classifier doCrossValidation(int seed, Instances trainingSet) 
            throws Exception {
        System.out.println();
        System.out.println("Doing Cross-Validation, SEED = " + seed + " ...");
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
        LibSVM mySVM, bestSVM;
        Instances train, test;
        Evaluation eval, evalAll;
        double best_Fbeta; // Weighted Harmonic Mean
        
        bestSVM    = null;
        best_Fbeta = 0.0;
        evalAll    = new Evaluation(randData);
        
        for ( int i = 0; i < NUM_FOLDS; i++ ) {
            eval  = new Evaluation(randData);
            train = randData.trainCV(NUM_FOLDS, i);
            test  = randData.testCV(NUM_FOLDS, i);
            
            // Build and evaluate classifier
            
            mySVM = new LibSVM(); // Construct SVM with Radial Basis Functions Kernel
            mySVM.setCost(SVM_COST);
            mySVM.setGamma(SVM_GAMMA);
            mySVM.setSeed(seed);
            mySVM.setProbabilityEstimates(true);
            mySVM.setNormalize(true);
            mySVM.buildClassifier(train);
            eval.evaluateModel(mySVM, test);
            evalAll.evaluateModel(mySVM, test);
            
            // Get best SVM based on Harmonic Mean (F-measure)
            if ( best_Fbeta < eval.weightedFMeasure() ) {
                best_Fbeta = eval.weightedFMeasure();
                bestSVM = (LibSVM) LibSVM.makeCopy(mySVM);
            }
            
            System.out.println("\n=== Fold " + (i+1) + "/" + NUM_FOLDS + 
                               ", Harmonic Mean (F-measure): "         + 
                               eval.weightedFMeasure() +" ===\n");
            
            //saveROCPlot(eval,i);
            
            /*
            // Output evaluation
            System.out.println();
            System.out.println("Harmonic Mean (F-measure): " + eval.weightedFMeasure() );
            System.out.println(eval.toMatrixString("=== Confusion matrix for fold " + 
                                                   (i+1) + "/" + NUM_FOLDS + " ===\n"));
            */
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

        //showROCPlot(NUM_FOLDS);
        
        // Show ROC Curve for each class
        for (int cl = 0; cl < NUM_CLASS; cl++) {
            showRocCurve(evalAll.predictions(), 
                         cl, 
                         randData.classAttribute().value(cl));
        }
        
        return bestSVM;
    }

    private static void showRocCurve(FastVector predictions, 
                                     int classIndex, String className) {
        // Generate Curve
        ThresholdCurve tc = new ThresholdCurve();
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
        
        jf = new javax.swing.JFrame("Weka Classifier Visualize: " + 
                                     className + " class");
        
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

    private static void getBestDataParameters() throws Exception {
        Instances data;
        Random randomGenerator = new Random();
        
        for (int numInstances : numInstances_list) {
            for (int numAttributes : numAttributes_list) {
                System.out.println("##########################################################");
                System.out.println("Training with " + numInstances   + 
                                   " Instances and " + numAttributes + 
                                   " Attributes");
                
                data = getTrainingSet(numInstances, numAttributes);
                doCrossValidation(randomGenerator.nextInt(999999), data);
            }
        }
    }

    private static void saveModel(Classifier classifier) {
        System.out.println("Saving SVM model to: " + MODEL + " ...");
        try {
            weka.core.SerializationHelper.write(MODEL, classifier);
        } catch (Exception ex) {
            System.out.println("\n=== Fail saving SVM model to: " + MODEL + " ===\n");
        }
    }
    
    private static Classifier getModel(){
        Classifier classifier = null;
        
        System.out.println("Getting SVM model from: " + MODEL + " ...");
        try {
            classifier = (Classifier) weka.core.SerializationHelper.read(MODEL);
        } catch (Exception ex) {
            System.out.println("=== Fail getting SVM model from: " + MODEL + " ===\n");
        }
        
        return classifier;
    }
/*
    private static void saveROCPlot(Evaluation eval, int i) throws Exception {
        ThresholdCurve tc = new ThresholdCurve();
        Instances result = tc.getCurve(eval.predictions(), 0);
        ArffSaver saver = new ArffSaver();
        
        saver.setInstances(result);
        saver.setFile(new File(OUT_PLOT + i +".arff"));
        saver.writeBatch();
    }
    
    private static void showROCPlot(int N) throws Exception {
        boolean first = true;
        ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
        
        for (int i = 0; i < N; i++) {
            Instances curve = DataSource.read(OUT_PLOT + i +".arff");
            curve.setClassIndex(curve.numAttributes() - 1);
            
            // method visualize
            PlotData2D plotdata = new PlotData2D(curve);
            plotdata.setPlotName("ROC_SVM "+ (i+1));
            plotdata.addInstanceNumberAttribute();
            
            // specify which points are connected
            boolean[] cp = new boolean[curve.numInstances()];
            for (int n = 1; n < cp.length; n++)
                cp[n] = true;
            
            plotdata.setConnectPoints(cp);
            // add plot
            
            if (first)
                tvp.setMasterPlot(plotdata);
            else
                tvp.addPlot(plotdata);
            
            first = false;
        }
     
        //Parametros Combobox
        tvp.setYIndex(7);
        tvp.setXIndex(8);
        tvp.setColourIndex(10);
        
        // method visualizeClassifierErrors
        final JFrame jf = new JFrame("WEKA ROC");
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(tvp, BorderLayout.CENTER);
        jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        jf.setVisible(true);
    }
*/
    private static void test(String file) {
       
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(file));
        } catch (FileNotFoundException ex) {
            System.err.println("File : " + file + " not found");
            System.exit(-1);
        }
        
        Instances testSet = mSVM_dataset.stringFreeStructure();
        
        while ( scanner.hasNextLine() ) {
            String tweet = scanner.nextLine();
            addInstance(tweet,testSet);
        }
        
        int numTweets = testSet.numInstances();
        for (int i = 0; i < numTweets; i++ ) {
            try {
                double prediction = mSVM.classifyInstance(testSet.instance(i));
                System.out.println(testSet.classAttribute().value((int) prediction));
            } catch (java.lang.Exception ex) {
                System.err.println("?: " + testSet.instance(i).toString());
            }
        }
    }
    
    private static void addInstance(String tweet, Instances data) {
        double[] values = new double[data.numAttributes()];
        Attribute wordAttribute;
        Scanner scanner = new Scanner(tweet);
        String word;
        
        while( scanner.hasNext() ) {
            word = scanner.next();
            wordAttribute = data.attribute(word);
            if ( wordAttribute != null ) {
                values[wordAttribute.index()] += 1.0;
            }
        }
        
        SparseInstance instance = new SparseInstance(1.0, values);
        instance.setMissing(0);

        data.add(instance);
    }
}

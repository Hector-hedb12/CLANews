package clanews.classifier;

import clanews.tweetsanalyzer.TweetsAnalyzer;
import java.awt.BorderLayout;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
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
    
    private static final String USAGE = "Usage: java -cp dist/CLANews.jar "  +
                                        "clanews.classifier.CLANews " +
                                        "-l <EN|ES> -m <TRAIN|TEST> "        +
                                        "[-i <test file> (if set TEST mode)]";
    // SVM Parameters
    private static final int BEST_NUM_INSTANCES  = 4000;
    private static final int BEST_NUM_ATTRIBUTES = 2500;
    private static final int BEST_SEED           = 691867;
    
    private static final int    NUM_FOLDS = 10;
    private static final double SVM_COST  = 4096.0;
    private static final double SVM_GAMMA = 6.103515625E-5;
    
    // Processed tweets
    private static final String DIR       = "News/Processed";
    private static final String TEST_FILE = "News/Test/tweets.txt";
    private static final String ARFF_FILE = "src/resources/classifier/all_tweets.arff";
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
    private static final String MODEL   = "src/resources/classifier/svm.model";
    private static final String DATASET = "src/resources/classifier/svm_model_dataset.arff";
    
    // Members variables
    private static Classifier mSVM;
    private static Instances mInstances;
    private static Instances mSVM_dataset;
    
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
    
    private static Instances getTrainingSet(int numInstances, 
                                            int numAttributes) 
            throws Exception {
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
    
    private static Classifier doCrossValidation(int seed, 
                                                Instances trainingSet,
                                                boolean showCharts)
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
        
        // Show ROC Curve for each class
        if ( showCharts ) {
            for (int cl = 0; cl < NUM_CLASS; cl++) {
                showRocCurve(evalAll.predictions(), 
                             cl, 
                             randData.classAttribute().value(cl));
            }
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
        ArffLoader loader = new ArffLoader();
        loader.setFile( new File(ARFF_FILE) );
        mInstances = loader.getDataSet();
        mInstances.setClassIndex(0);
        
        Instances data;
        Random randomGenerator = new Random();
        
        for (int numInstances : numInstances_list) {
            for (int numAttributes : numAttributes_list) {
                System.out.println("##########################################################");
                System.out.println("Training with "  + numInstances  + 
                                   " Instances and " + numAttributes + 
                                   " Attributes");
                
                data = getTrainingSet(numInstances, numAttributes);
                doCrossValidation(randomGenerator.nextInt(999999), data, false);
            }
        }
    }
    
    private static void getBestSeedParameter() throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setFile( new File(ARFF_FILE) );
        mInstances = loader.getDataSet();
        mInstances.setClassIndex(0);
        
        Instances data = getTrainingSet(BEST_NUM_INSTANCES, BEST_NUM_ATTRIBUTES);
        Random randomGenerator = new Random();
        
        for (int i = 0; i < 25; i++) {
            doCrossValidation(randomGenerator.nextInt(999999), data, false);
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

    private static void test(String file) {
        System.out.println("Testing model over: " + file);
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(file));
        } catch (FileNotFoundException ex) {
            System.err.println("File : " + file + " not found");
            System.exit(-1);
        }
        
        Instances testSet = mSVM_dataset.stringFreeStructure();
        TweetsAnalyzer myAnalyzer = new TweetsAnalyzer("EN");
        
        while ( scanner.hasNextLine() ) {
            String tweet = scanner.nextLine();
            addInstance(myAnalyzer.analyze(tweet),testSet);
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
        System.out.println("\tFinished");
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
    
    /**
     * @param options the command line arguments
     */
    public static void main(String[] options) {
        String language  = "";
        String test_file = "";
        boolean testMode = false;
        
        // Parse options
        try {
            String mode = "";
            
            language = Utils.getOption('l', options);
            mode     = Utils.getOption('m', options);
            
            if ( !language.equals("EN") && !language.equals("ES") ) {
                throw new Exception("Supported languages: EN or ES");
            }
            
            if ( !mode.equals("TRAIN") && !mode.equals("TEST") ) {
                throw new Exception("Supported modes: TRAIN or TEST");
            }

            testMode = mode.equals("TEST");
            test_file = Utils.getOption('i', options);
            
            // Read test input file
            if ( testMode ) {
                if ( test_file.equals("") ) {
                    test_file = TEST_FILE;
                }
            }
            
        } catch (Exception ex) {
            System.err.println(ex.toString());
            System.err.println(USAGE);
            System.exit(-1);
        }
        
        // Execute
        ArffLoader loader = new ArffLoader();
        
        if ( testMode ) {
            mSVM = getModel(); // try get model
        }
        
        if ( mSVM == null) {
            if ( testMode ) {
                System.out.println("Making classifier before test it ...");
            }
            
            // Load all tweets
            try {
                loader.setFile( new File(ARFF_FILE) );
                mInstances = loader.getDataSet();
                System.out.println("Loading ARFF File...");
            } catch (NullPointerException ex) { // File not found
                try {
                    if ( mInstances == null ) {
                        fromTweetsToArff(DIR, ARFF_FILE);
                    }
                } catch (Exception ex1) {
                    System.err.println("Reading Training Set from dir: " + DIR);
                    System.exit(-1);
                }
            } catch (Exception ex) {
                System.err.println("Getting Training Set");
                System.exit(-1);
            }

            mInstances.setClassIndex(0);
            
            try {
                // Get sample of all tweets
                mSVM_dataset = getTrainingSet(BEST_NUM_INSTANCES,
                                              BEST_NUM_ATTRIBUTES);
                mSVM_dataset.setRelationName("svm_model_dataset");
            } catch (Exception ex) {
                System.err.println("Getting Sample of Training Set");
                System.exit(-1);
            }
            
            try {
                // Train and Save SVM
                mSVM = doCrossValidation(BEST_SEED, mSVM_dataset, !testMode);
                saveModel( mSVM );
            } catch (Exception ex) {
                System.err.println("Training or Saving Classifier");
                System.exit(-1);
            }
            
            try {
                // Save dataset used to train SVM
                ArffSaver saver = new ArffSaver();
                saver.setInstances(mSVM_dataset);
                saver.setFile( new File(DATASET) );
                saver.writeBatch();
            } catch (IOException ex) {
                System.err.println("Saving dataset used to train SVM");
                System.exit(-1);
            }
        }
        
        if ( mSVM_dataset == null ) {
            // Load dataset used to train SVM
            try {
                loader.setFile( new File(DATASET) );
                mSVM_dataset = loader.getDataSet();
                mSVM_dataset.setClassIndex(mSVM_dataset.numAttributes()-1);
            } catch (Exception ex) { // File not found
                System.err.println("Loading ARFF File: " + DATASET);
                System.exit(-1);
            }
        }
        
        if ( testMode ) {
            test(test_file);
        }

/*
        // Tuning data parameters:
        try {
            getBestDataParameters();
        } catch (Exception ex) {
            System.err.println("Getting Best Parameters");
            System.exit(-1);            
        }

        // Tuning seed:
        try {
            getBestSeedParameter();
        } catch (Exception ex) {
            System.err.println("Getting Best Seed Parameter");
            System.exit(-1);            
        }
*/
    }
}

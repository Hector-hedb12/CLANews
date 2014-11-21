/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
    
    private static final String DIR = "../News/Processed";
    private static final String OUT = "src/resources/SVM_input.arff";
    
    private static Instances mInstances, mStructure;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setFile( new File(OUT) );
        
        //mStructure = loader.getStructure();
        mInstances = loader.getDataSet();
        
        if ( mInstances == null ) {
            System.out.println("Reading Tweets from " + DIR + " directory...");
            fromTweetsToArff(DIR, OUT);
        }
        
        mInstances.setClassIndex(0);
        
        train(1000, 1000);
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
        //System.out.println("\n\nFiltered data:\n\n" + data);
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(mInstances);
        saver.setFile( new File(outputFile) );
        saver.writeBatch();
    }
    
    private static void train(int numInstances, int numAttributes) throws Exception {
        String[] options;
        Instances newData;
        
        // Get a sample of all instances
        options = new String[] {"-M", "1.0",
                                "-X", Integer.toString(numInstances),
                                "-S", "1" };
        
        SpreadSubsample subSampleFilter = new SpreadSubsample();
        subSampleFilter.setOptions(options);
        subSampleFilter.setInputFormat(mInstances);
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
        newData = Filter.useFilter(newData, attributeFilter);
        
        // Construct SVM
        SMO svm = new SMO();
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
        
        Evaluation eval = new Evaluation(newData);
        eval.crossValidateModel(svm, newData, NUM_FOLDS, new Random(1));
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        //System.out.println(svm.toString());
        
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

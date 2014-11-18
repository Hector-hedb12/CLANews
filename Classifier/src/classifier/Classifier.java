/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Hector E. Dominguez B.
 */
public class Classifier {
    private static final String DIR = "../News/Processed";
    private static final String OUT = "src/resources/SVM_input.arff";
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(new File(DIR));
        Instances rawData = loader.getDataSet();
        //System.out.println("\n\nRaw data:\n\n" + rawData);
                
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(rawData);
        //filter.setOutputWordCounts(true);
        //filter.setWordsToKeep(1000000);
        //filter.setDoNotOperateOnPerClassBasis(true);
        Instances data = Filter.useFilter(rawData, filter);
        //System.out.println("\n\nFiltered data:\n\n" + data);
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile( new File(OUT) );
        saver.writeBatch();
    }
}

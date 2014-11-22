package tweetsanalyzer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.es.SpanishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.WordlistLoader;

/**
 *
 * @author Hector E. Dominguez B.
 */
public class TweetsAnalyzer {

    private static final String ES_FILE = "resources/spanish_stopwords.txt";
    private static final String EN_FILE = "resources/english_stopwords.txt";
    private static final String RAW_DIR = "../News/Raw";
    private static final String PRO_DIR = "../News/Processed";
    
    private static CharArraySet mStopWords;
    private static Analyzer mAnalyzer;
    
    public TweetsAnalyzer(String language) {
        // Read language
        if ( language.equalsIgnoreCase("ES") ) {
            init(ES_FILE);
        } else if ( language.equalsIgnoreCase("EN") ) { 
            init(EN_FILE);
        } else {
            System.err.println("Language not supported yet");
            System.exit(-1);
        }
    }
    
    /* Init member variables */
    private void init(String stopWordsFile) {
        try {
            //System.out.println(this.getClass().getClassLoader().getResource(EN_FILE).getFile());
            ClassLoader classloader = getClass().getClassLoader();
            File wordsFile = new File(classloader.getResource(stopWordsFile).getFile());
            mStopWords = WordlistLoader.getWordSet( new FileReader(wordsFile) );
        } catch (FileNotFoundException ex) {
            System.err.println("File: " + stopWordsFile + " not found");
            System.exit(-1);
        } catch (IOException ex) {
            System.err.println("Reading file: " + stopWordsFile);
            System.exit(-1);
        }
        
        if ( stopWordsFile.equals(EN_FILE) ) {
            mAnalyzer = new EnglishAnalyzer(mStopWords);
        } else {
            mAnalyzer = new SpanishAnalyzer(mStopWords);
        }
    }

    /* The workflow of the TokenStream API is as follows:
     * 1 - Instantiation of TokenStream/TokenFilters which add/get attributes 
     *     to/from the AttributeSource.
     * 2 - The consumer calls reset().
     * 3 - The consumer retrieves attributes from the stream and stores local 
     *     references to all attributes it wants to access.
     * 4 - The consumer calls incrementToken() until it returns false consuming 
     *     the attributes after each call.
     * 5 - The consumer calls end() so that any end-of-stream operations can be 
     *     performed.
     * 6 - The consumer calls close() to release any resource when finished 
     *     using the TokenStream.
     *
     * This method internally:
     * a) Use a grammar-based tokenizer constructed with JFlex that:
     *    - Splits words at punctuation characters, removing punctuation. 
     *      However, a dot that's not followed by whitespace is considered 
     *      part of a token.
     *    - Splits words at hyphens, unless there's a number in the token, in 
     *      which case the whole token is interpreted as a product number and 
     *      is not split.
     *    - Recognizes email addresses and internet hostnames as one token. 
     * b) Normalizes tokens extracted with StandardTokenizer: remove 's and 
     *    remove dots
     * c) Normalizes token text to lower case
     * d) Removes stop words from a token stream
     * e) Transforms the token stream as per the Porter stemming algorithm
     */    
    public String analyze(String sentence) { 
        StringBuilder sb = new StringBuilder();
        
        try {
            TokenStream ts = mAnalyzer.tokenStream("myfield", 
                                                  new StringReader(sentence));  // 1
            OffsetAttribute offsetAtt = ts.addAttribute(OffsetAttribute.class); // 1
            
            ts.reset();                                                         // 2
            CharTermAttribute charTermAttr = 
                    ts.getAttribute(CharTermAttribute.class);                   // 3
            
            while (ts.incrementToken()) {                                       // 4
                sb.append(charTermAttr.toString());
                sb.append(" ");
            }
            
            ts.end();                                                           // 5
            ts.close();                                                         // 6
        } catch (IOException ex) {
            System.err.println("Analyzing...");
            System.exit(-1);
        }
        
        return sb.toString();
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
/*
        TweetsAnalyzer myAnalyzer = new TweetsAnalyzer("EN");
        File raw_dir = new File(RAW_DIR);
        File [] categories = raw_dir.listFiles();
        File [] tweets;
        
        for ( File category : categories ) {
            
            if ( !category.isDirectory() ) {
                continue;
            }
            
            System.out.println(category.getName());
            tweets = ( new File(RAW_DIR + "/" + category.getName()) ).listFiles();
            for ( File tweet : tweets ) {
                System.out.println(tweet.getName());
            }
        }
        //System.out.println(mt.getClass().getClassLoader().getResource(EN_FILE).getFile());
*/
    }
}

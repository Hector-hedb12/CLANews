package tweetsgetter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.es.SpanishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.WordlistLoader;
import twitter4j.*;

/**
 *
 * @author Hector E. Dominguez B.
 */
public class TweetsGetter {
    
    private static final String ES_FILE  = "src/resources/spanish_stopwords.txt";
    private static final String EN_FILE  = "src/resources/english_stopwords.txt";
    private static final String INPUT    = "src/resources/input.txt";
    private static final int N_TWEETS    = 5000;
    private static final int EXTRA_PAGES = 6;
    
    private static final String URL_REGEX = "\\(?\\b((http|https)://|www[.])[-A-Za-z0-9+&amp;@#/%?=~_()|!:,.;]*[-A-Za-z0-9+&amp;@#/%=~_()|]";
    
    private static CharArraySet mStopWords;
    private static Analyzer mAnalyzer;
    private static Twitter mTwitter;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // Read language
        if ( args.length == 1 && args[0].equalsIgnoreCase("ES") ) {
            init(ES_FILE);
        } else { 
            // Default
            init(EN_FILE);
        }
       
        // Read input file
        ArrayList<String> inputList = readInputFile();
        
        List<Status> statuses; // tweets list
        Paging paging;         // result paging
        String tweet;          // tweet text
        int tweets_per_user;   // number of tweet per user
        int rest;              // number of tweet for last user
        int count;             // count tweets per user
        int n_users;           // number of users
        int n_pages;           // number of pages
        
        BufferedWriter out;
        String outputFileName;
        
        // Read each line
        File directory;
        ArrayList<String> userList;
        for ( String line : inputList ){
            userList = readLine( line );
            
            // Create sub-directory to output
            directory = new File( userList.get(0) );
            if ( !directory.exists() ) {
                directory.mkdirs();
            }
            
            n_users = userList.size() - 2;
            tweets_per_user = N_TWEETS / n_users;
            rest = (N_TWEETS % n_users);
            n_pages = (tweets_per_user + rest) / 200 + EXTRA_PAGES; // Worst case
            
            paging = new Paging(n_pages, 200);
            
            System.out.println("tweets per user: " + tweets_per_user    + 
                    ", last user: " + (tweets_per_user + rest)          + 
                    ", paging numbers: " + n_pages);
            
            for( int user = 2; user < userList.size(); user++) {
                outputFileName = userList.get(0) + userList.get(1) + (user-1) + ".txt";
                out = getOutputFile( outputFileName );
                
                System.out.println("Writing: " + outputFileName);
                
                if ( user == userList.size() - 1 ) {
                    tweets_per_user += rest;
                }
                
                count = 0;
                // Read and write to output file user's tweets
                for (int page = 1; page <= n_pages && count < tweets_per_user; page++) {
                    try {
                        paging.setPage(page);
                        statuses = mTwitter.getUserTimeline(userList.get(user), 
                                                            paging );
                        for (Status status : statuses) {
                            
                            if ( count == tweets_per_user ) {
                                break;
                            }
                            
                            tweet = status.getText();
                            // Not Retweet
                            if ( !tweet.contains("RT @") ) {
                                tweet = tweet.replaceAll(URL_REGEX, "");
                                writeTweet(out, analyze(mAnalyzer, tweet) );
                                count++;

                                if ( count % 100 == 0 ) {
                                    System.out.println("Writing tweet: " + count);
                                }
                            }
                        }
                    } catch (TwitterException ex) {
                        ex.printStackTrace();
                        System.err.println("Getting Timeline (user, paging): (" + (user-1) + "," + paging + ")");
                        System.exit(-1);
                    }
                }
                
                try {
                    out.close();
                } catch (IOException ex) {
                    System.err.println("Closing: " + out.toString() );
                    System.exit(-1);
                }
            }
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
    private static String analyze(Analyzer analyzer, String sentence) { 
        StringBuilder sb = new StringBuilder();
        
        try {
            TokenStream ts = analyzer.tokenStream("myfield", 
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

    /* Init member variables */
    private static void init(String stopWordsFile) {
        try {
            mStopWords = WordlistLoader.getWordSet( new FileReader(stopWordsFile) );
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
        
        mTwitter = new TwitterFactory().getInstance();
    }

    /* Read a small input file into list */
    private static ArrayList<String> readInputFile() {
        Scanner scanner;
        ArrayList<String> inputList = new ArrayList<String>();
        
        try {
            scanner = new Scanner( new File(INPUT) ).useDelimiter("\n");
            while ( scanner.hasNext() ) {
                inputList.add( scanner.next() );
            }
        } catch (FileNotFoundException ex) {
            System.err.println("File : " + INPUT + " not found");
            System.exit(-1);
        }
        
        return inputList;
    }
    
    /* Read a line into list */
    private static ArrayList<String> readLine(String line) {
        ArrayList <String> userList = new ArrayList<String>();
        Scanner scanner = new Scanner(line);
        
        while( scanner.hasNext() ) {
            userList.add( scanner.next() );
        }
        
        return userList;
    }

    /* Get BufferedWriter to write output */
    private static BufferedWriter getOutputFile(String fileName) {
        try {
            FileWriter writer;
            BufferedWriter out;
            writer = new FileWriter( fileName );
            out = new BufferedWriter(writer);
            return out;
        } catch (IOException ex) {
            System.err.println("Creating File : " + fileName);
            System.exit(-1);
        }
        
        return null;
    }

    /* Write a tweet in a BufferedWriter */
    private static void writeTweet(BufferedWriter out, String tweet) {
        try {
            out.write(tweet);
            out.newLine();
        } catch (IOException ex) {
            System.err.println("Writing tweet: " + tweet);
            System.exit(-1);
        }
    }
}

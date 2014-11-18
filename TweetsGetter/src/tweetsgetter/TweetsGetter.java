package tweetsgetter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import tweetsanalyzer.TweetsAnalyzer;
import twitter4j.*;

/**
 *
 * @author Hector E. Dominguez B.
 */
public class TweetsGetter {
    
    private static final String INPUT    = "src/resources/input.txt";
    private static final String RAW_DIR  = "Raw/";
    private static final String PRO_DIR  = "Processed/";
    private static final int N_TWEETS    = 5000;
    private static final int EXTRA_PAGES = 6;
    
    private static final String URL_REGEX = "\\(?\\b((http|https)://|www[.])[-A-Za-z0-9+&amp;@#/%?=~_()|!:,.;]*[-A-Za-z0-9+&amp;@#/%=~_()|]";
    private static final String NUM_REGEX = "\\d+";
    
    private static Twitter mTwitter;
    private static TweetsAnalyzer mAnalyzer;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        init(); // Init member variables
        
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
                
        String rawSubDir, proSubDir;
        String rootDir = inputList.remove(0);
        createDirectory(rootDir);
        
        // Read each user
        ArrayList<String> userList;
        for ( String line : inputList ) {
            
            userList = readLine( line );
            rawSubDir = rootDir + RAW_DIR + userList.get(0);
            proSubDir = rootDir + PRO_DIR + userList.get(0);
            createDirectory(rawSubDir);
            createDirectory(proSubDir);
            
            n_users = userList.size() - 1;
            tweets_per_user = N_TWEETS / n_users;
            rest = (N_TWEETS % n_users);
            n_pages = (tweets_per_user + rest) / 200 + EXTRA_PAGES; // Worst case
            
            paging = new Paging(n_pages, 200);
            
            System.out.println("tweets per user: " + tweets_per_user      + 
                               ", last user: " + (tweets_per_user + rest) + 
                               ", paging numbers: " + n_pages);
            
            for( int user = 1; user < userList.size(); user++) {
                
                if ( user == userList.size() - 1 ) {
                    tweets_per_user += rest;
                }
                
                count = 1;
                // Read and write to output file user's tweets
                for (int page = 1; page <= n_pages && count <= tweets_per_user; page++) {
                    try {
                        paging.setPage(page);
                        statuses = mTwitter.getUserTimeline(userList.get(user), 
                                                            paging );
                        for (Status status : statuses) {
                            
                            if ( count > tweets_per_user ) {
                                break;
                            }
                            
                            tweet = status.getText();
                            // Not Retweet
                            if ( !tweet.contains("RT @") ) {
                                writeTweet(rawSubDir + userList.get(user) + count + ".txt", 
                                           tweet);
                                
                                tweet = tweet.replaceAll(URL_REGEX, "");
                                tweet = tweet.replaceAll(NUM_REGEX, "");
                                
                                writeTweet(proSubDir + userList.get(user) + count + ".txt" , 
                                           mAnalyzer.analyze(tweet) );
                                
                                count++;

                                if ( count % 100 == 0 ) {
                                    System.out.println("Writing tweet: " + userList.get(user) + count + ".txt");
                                }
                            }
                        }
                    } catch (TwitterException ex) {
                        ex.printStackTrace();
                        System.err.println("Getting Timeline (user, paging): (" + (user-1) + "," + paging + ")");
                        System.exit(-1);
                    }
                }
            }
        }
    }

    /* Init member variables */
    private static void init() {
        mTwitter  = new TwitterFactory().getInstance();
        mAnalyzer = new TweetsAnalyzer("EN");
    }
    
    private static void createDirectory(String name) {
        File directory = new File( name );
        if ( !directory.exists() ) {
            directory.mkdirs();
        }
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

    /* Write a tweet in a BufferedWriter */
    private static void writeTweet(String fileName, String tweet) {
        try {
            // Get BufferedWriter to write output
            FileWriter writer;
            BufferedWriter out;
            writer = new FileWriter( fileName );
            out = new BufferedWriter(writer);
            // Write tweet
            out.write(tweet);
            // Close BufferedWriter
            out.close();
        } catch (IOException ex) {
            System.err.println("Writing tweet: " + fileName);
            System.exit(-1);
        }
    }
}

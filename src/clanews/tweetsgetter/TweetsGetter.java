package clanews.tweetsgetter;

import clanews.tweetsanalyzer.TweetsAnalyzer;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import twitter4j.*;
import weka.core.Utils;

/**
 *
 * @author Hector E. Dominguez B.
 */
public class TweetsGetter {
    
    private static final String USAGE = "Usage: java -cp dist/CLANews.jar " +
                                        "clanews.tweetgetter.TweetsGetter " +
                                        "-l <EN|ES> -m <TRAIN|TEST> "       +
                                        "-n <number of tweets> [-i <input file>]";
    
    private static final String TRAIN_INPUT = "src/resources/tweetsgetter/train_input.txt";
    private static final String TEST_INPUT  = "src/resources/tweetsgetter/test_input.txt";
    
    private static final String RAW_DIR  = "Raw/";
    private static final String PRO_DIR  = "Processed/";
    
    private static final int EXTRA_PAGES = 6;
    
    private static Twitter mTwitter;
    private static TweetsAnalyzer mAnalyzer;
    
    private static void createDirectory(String name) {
        File directory = new File( name );
        if ( !directory.exists() ) {
            directory.mkdirs();
        }
    }
    
    /* Read a small input file into list */
    private static ArrayList<String> readInputFile(String train_input) {
        Scanner scanner;
        ArrayList<String> inputList = new ArrayList<String>();
        
        try {
            scanner = new Scanner( new File(train_input) ).useDelimiter("\n");
            while ( scanner.hasNext() ) {
                inputList.add( scanner.next() );
            }
        } catch (FileNotFoundException ex) {
            System.err.println("File : " + train_input + " not found");
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
    
    private static void getAndSaveTrainingTweets(ArrayList<String> inputList, 
                                                 int n_tweets) {
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
            tweets_per_user = n_tweets / n_users;
            rest = (n_tweets % n_users);
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
    
    private static void getAndSaveTestingTweets(ArrayList<String> inputList,
                                                int n_tweets) {
        List<Status> statuses; // tweets list
        Paging paging;         // result paging
        String tweet;          // tweet text
        int count;             // count tweets per user
        int n_pages;           // number of pages

        String rootDir = inputList.remove(0);
        createDirectory(rootDir);
        
        // Get BufferedWriter to write output
        BufferedWriter out = null;
        try {
            out = new BufferedWriter(new FileWriter( rootDir + "tweets.txt" ));
        } catch (IOException ex) {
            System.err.println("Opening File: " + rootDir + "tweets.txt");
            System.exit(-1);
        }
        
        // Read each user
        for ( String user : inputList ) {
            n_pages = n_tweets / 200 + EXTRA_PAGES; // Worst case
            
            paging = new Paging(n_pages, 200);
            
            System.out.println("tweets per user: "  + n_tweets + 
                               ", paging numbers: " + n_pages);
            
            count = 1;
            // Read and write to output file user's tweets
            for (int page = 1; page <= n_pages && count <= n_tweets; page++) {
                try {
                    paging.setPage(page);
                    statuses = mTwitter.getUserTimeline(user, paging );
                    for (Status status : statuses) {

                        if ( count > n_tweets ) {
                            break;
                        }

                        tweet = status.getText();
                        // Not Retweet
                        if ( !tweet.contains("RT @") ) {
                            out.write(tweet);
                            out.newLine();
                            count++;

                            if ( count % 100 == 0 ) {
                                System.out.println("Writing tweet: " + count);
                            }
                        }
                    }
                } catch (TwitterException ex) {
                    ex.printStackTrace();
                    System.err.println("Getting Timeline (user, paging): (" + (user) + "," + paging + ")");
                    System.exit(-1);
                } catch (IOException ex) {
                    System.err.println("Writing tweets");
                    System.exit(-1);
                }
            }
        }
        
        try {
            out.close();
        } catch (IOException ex) {
            System.err.println("Closing File: " + rootDir + "tweets.txt");
            System.exit(-1);
        }
    }
    
    /**
     * @param options the command line arguments
     */
    public static void main(String[] options) {
        String language   = "";
        String mode       = "";
        String input_file = "";
        int n_tweets    = 0;
        
        try {
            language = Utils.getOption('l', options);
            mode     = Utils.getOption('m', options);
            n_tweets = Integer.parseInt( Utils.getOption('n', options) );
            
            if ( !language.equals("EN") && !language.equals("ES") ) {
                throw new Exception("Supported languages: EN or ES");
            }
            
            if ( !mode.equals("TRAIN") && !mode.equals("TEST") ) {
                throw new Exception("Supported modes: TRAIN or TEST");
            }
            
            if ( 5000 < n_tweets || n_tweets < 100 ) {
                throw new Exception("Number of tweets must be in: [100, 5000]");
            }
         
            // Init member variables
            mTwitter  = new TwitterFactory().getInstance();
            mAnalyzer = new TweetsAnalyzer(language);
            
            input_file = Utils.getOption('i', options);
            
            // Read train input file
            ArrayList<String> inputList;
            if ( !input_file.equals("") ) {
                inputList = readInputFile( input_file );
            } else {
                if ( mode.equals("TRAIN") ) {
                    inputList = readInputFile(TRAIN_INPUT);
                } else {
                    inputList = readInputFile(TEST_INPUT);
                }
            }
            
            if ( mode.equals("TRAIN") ) {
                getAndSaveTrainingTweets(inputList, n_tweets);
            } else {
                getAndSaveTestingTweets(inputList, n_tweets);
            }
            
        } catch (Exception ex) {
            System.err.println(ex.toString());
            System.err.println(USAGE);
            System.exit(-1);
        }
    }
}
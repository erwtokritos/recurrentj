/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package R.tests;

import R.Graph;
import R.Mat;
import R.Solver;
import R.Utils;
import R.objects.BigFile;
import R.objects.Cost;
import R.objects.Generator;
import R.objects.Model;
import R.objects.SolverStats;
import R.objects.State;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 *
 * @author thanos
 */
public class LetterEmbeddingExample {

    static HashMap<String, Integer> letterToIndex = new HashMap<>(2000);
    static HashMap<Integer, String> indexToLetter = new HashMap<>(2000);

    //model parameters
    static Model model = null;
    static Solver s = new R.Solver();
    static Generator generator = Generator.LSTM;  // can be 'rnn' or 'lstm'
    static int[] hidden_sizes = new int[]{128, 128}; //list of sizes of hidden layers
    static int letter_size = 20;                   // size of letter embeddings
    static int input_size;
    static int output_size;
    static int epoch_size;
    static List<String> vocab;
    static int noOfIterations = 1000;

    //optimization
    static double regc = 0.000001;        // L2 regularization strength
    static double learning_rate = 0.01; // learning rate
    static double clipval = 5.0;          // clip gradients at this value    

    //data
    static List<String> data_sents;
    static Random random = new Random(new Date().getTime());
    static String filePath = "datasets/text/PaulGraham.txt";
    
    //stats..
    static List<Double> ppl_list = new ArrayList<>();
    static int tickIter = 0;
    
    //prediction
    //static boolean samplei = false;
    static double sample_softmax_temperature = 3.0;
    static int max_chars_gen = 100;
    
    
    
    
    public static void main(String[] args) throws Exception {

        loadData();
        initVocab();
        
        input_size = vocab.size() + 1;
        output_size = vocab.size() + 1;
        epoch_size = data_sents.size();
        
        System.out.println("#input size = " + input_size);
        System.out.println("#output size = " + output_size);
        System.out.println("#epoch size = " + epoch_size);
        
        initModel();
        
        for(int iteration = 0; iteration < noOfIterations*epoch_size; iteration++) {
            tick();
        }
        
        


    }
    
       
    public static String predictSentence(boolean samplei) throws Exception {
        
        
        Graph G = new Graph(false);
        String s = "";
        State prev = new State();
        
        while(true) {
            
            //RNN tick
            int ix = (s.length() == 0) ? 0 : letterToIndex.get(s.substring(s.length() - 1));
            State lh = forwardIndex(G, model, ix, prev);
            prev = lh;
            
            
            //sample predicted letter
            Mat logprobs = lh.o;
            if(sample_softmax_temperature != 1.0 && samplei) {
                // scale log probabilities by temperature and renormalize
                // if temperature is high, logprobs will go towards zero
                // and the softmax outputs will be more diffuse. if temperature is
                // very low, the softmax outputs will be more peaky
                for(int q = 0; q < logprobs.w.length; q++) {
                    logprobs.w[q] /= sample_softmax_temperature;
                }
            }
            
            
            Mat probs = Utils.softmax(logprobs);            
            if(samplei) {
                ix = Utils.samplei(probs.w);                
            }else  {
                ix = Utils.maxi(probs.w);
            }
            
            if(ix == 0) {  //END TOKEN predicted, break out
                break;
            }
            
            if(s.length() > max_chars_gen) { //something is wrong
                break;
            }
            
            
            String letter = indexToLetter.get(ix);
            s += letter;
            
        }
        
        return s;
        
    }
    
    public static void initModel() {
                     
        if (generator.equals(Generator.LSTM)) {
            model = Utils.initLSTM(letter_size, hidden_sizes, output_size);
        } else {
            model = Utils.initRNN(letter_size, hidden_sizes, output_size);
        }

        //add letter embedding matrix
        model.put("Wil", Utils.RandMat(input_size, letter_size, 0.0, 0.08));
    }

    private static void initVocab() {
        
        vocab = new ArrayList<>();
        
        int index = 0;
        for(String line : data_sents) {
            
            for(int i = 0; i < line.length(); i++) {
                
                String character = String.valueOf(line.charAt(i));
                if(!letterToIndex.containsKey(character)) {
                    
                    letterToIndex.put(character, index);
                    indexToLetter.put(index, character);
                    vocab.add(character);
                    index++;
                }
            }
            
        }
        
        System.out.println("There are " + letterToIndex.keySet().size() + " distinct letters..");
    }
    
    
    public static void tick() throws Exception {
        
        //sample sentence from data
        int sentix = random.nextInt(data_sents.size());
        //int sentix = tickIter % data_sents.size();
        String sent = data_sents.get(sentix);
        
        Date t0 = new Date();
        Cost cost = costfun(model, sent);
        Graph G = cost.G;        
        G.backward();        
        SolverStats solver_stats = s.step(model, learning_rate, regc, clipval);
         
        Date t1 = new Date();
        long tick_time = t1.getTime() - t0.getTime();
        ppl_list.add(cost.ppl);
        
        tickIter++;
        
        if(tickIter % (epoch_size/2) == 0) {
            String pred = predictSentence(true);
            System.out.println("Predicted sentence : " + pred);
        }
        
        if(tickIter % epoch_size == 0) {
            double median_ppl = Utils.median(ppl_list);
            ppl_list.clear();
            System.out.println("Perplexity : " + median_ppl);
        }
        
    }
    
    
    public static Cost costfun(Model model, String sent) throws Exception {

        int n = sent.length();
        Graph G = new Graph();
        double log2ppl = 0.0;
        double cost = 0.0;

        State prev = null;
        for (int i = -1; i < n; i++) {

            // start and end tokens are zeros
            // first step: start with START token
            int ix_source = (i == -1) ? 0 : letterToIndex.get(sent.substring(i, i + 1));
            // last step : end with END token
            int ix_target = (i == n - 1) ? 0 : letterToIndex.get(sent.substring(i + 1, i + 2));

            State lh = forwardIndex(G, model, ix_source, prev);
            prev = lh;

            //set gradients into logprobabilities
            Mat logprobs = lh.o;  // interpret output as logprobs
            Mat probs = Utils.softmax(logprobs); // compute the softmax probabilities

            log2ppl += -Utils.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
            cost += -Math.log(probs.w[ix_target]);

            //write gradients into log probabilities
            logprobs.dw = probs.w;
            logprobs.dw[ix_target] -= 1.0;

        }

        double ppl = Math.pow(2, log2ppl / (n - 1));

        Cost c = new Cost();
        c.G = G;
        c.cost = cost;
        c.ppl = ppl;
        
        return c;
    }
    
    private static void loadData() throws Exception {
        
        data_sents = new ArrayList<>();
        BigFile file = new BigFile(filePath);
        Iterator<String> it = file.iterator();
        while(it.hasNext()) {
            String line = it.next();
            data_sents.add(line);
        }
        
        System.out.println("#Read " + data_sents.size() + " lines..");
    }

    private static State forwardIndex(Graph G, Model model, int ix, State prev) throws Exception {

        Mat x = G.rowPluck(model.get("Wil"), ix);

        if (generator.equals(Generator.RNN)) {
            return Utils.forwardRNN(G, model, hidden_sizes, x, prev);
        } else {
            return Utils.forwardLSTM(G, model, hidden_sizes, x, prev);
        }
    }

}

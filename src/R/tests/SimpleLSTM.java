/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package R.tests;

import R.Graph;
import R.Mat;
import R.Utils;
import R.objects.State;
import R.objects.Model;

/**
 *
 * @author thanos
 */
public class SimpleLSTM {
    
    public static void main(String[] args) throws Exception {
        
        // takes as input Mat of 10x1, contains 2 hidden layers of
        // 20 neurons each, and outputs a Mat of size 2x1
        int[] hidden_sizes = new int[]{20, 20};
        Model lstm_model = Utils.initLSTM(10, hidden_sizes, 2);
        
        Mat x1 = Utils.RandMat(10, 1); // example input #1
        Mat x2 = Utils.RandMat(10, 1); // example input #2
        Mat x3 = Utils.RandMat(10, 1); // example input #3
        
        Graph G = new Graph();
        State out1 = Utils.forwardLSTM(G, lstm_model, hidden_sizes, x1, new State());                
        State out2 = Utils.forwardLSTM(G, lstm_model, hidden_sizes, x2, out1);            
        State out3 = Utils.forwardLSTM(G, lstm_model, hidden_sizes, x3, out2);
        
         // the field.o contains the output Mats:
        // e.g. x1.o is a 2x1 Mat
        // for example lets assume we have binary classification problem
        // so the output of the LSTM are the log probabilities of the
        // two classes. Lets first get the probabilities:
        Mat probs = Utils.softmax(out1.o);
        int ix_target = 0; // suppose first input has class 0
        double cost =  -Math.log(probs.w[ix_target]);        
        System.out.println(cost);

        G.backward();     
    }
}

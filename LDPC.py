import numpy as np

class LDPC:
    '''
    LDPC decoder class
    
    Inputs:
        H    - Parity Check Matrix
        y    - Message received from channel
        eps  - Error rate
        runs - (Optional) Number of iterations to run loopy BP, default to 20 runs
        
    Public Methods:
        message_passing - Calling message_passing() will perform loopy BP until convergence
                          or max number of iterations of performed.
    Output:
        xpred - After running message_passing, variable xpred is available which our best
                prediction of codeword sent across channel.
    '''
    def __init__(self,H,y,eps,runs=20):
        
        # Inputs
        self.H = H
        self.y = y
        self.eps = eps
        self.runs = runs
        
        # Initialize helper variables
        self._initialize()
        
    def _initialize(self):
        '''
        Hidden function to initialize the helper variables required to perform loopy BP in our
        decoding.
        '''
        # Keep track of convergence of message passing algorithm
        self.converged = False
        # Track number of iterations of loopy BP performed
        self.l = 0
        
        # We will work with likelihood ratios (LRS). Symbol wise if a 0 is recvd then
        # set the lik ratio to ln((1-eps)/eps) else ln(eps/(1-eps)) for the channel LRS.
        self.ch_liks = np.log([ ((1-eps)/eps) if i==0 else (eps/(1-eps)) for i in self.y])
        # Initialize variable likelihoods to likelihoods from the channel.
        self.var_liks = self.ch_liks.copy()
        
        # CHECK2VARIABLE MESSAGES
        # Create two dictionaries
        # 1. map each check node index to its neighbouring variable node indexes
        # 2. map each check node index to a dictionary with variable node index to check2variable 
        #    message value - initialize check2variable messages to 0.
        
        # For check2variable mapping multiply each row with 1-N to fill non-zero cells with an index value
        # subtracting 1 will set 0 value cells to -1 and allow us to use pythons 0 val indexing.
        H1 = self.H*range(1, self.H.shape[1]+1)-1
        # Create check2variable index dict, ignore -1 values as imply no connection
        self.c2v_idx = {i: list(h[h!=-1.]) for i,h in enumerate(H1) }
        # Create check2variable message dict, initialize messages to 0
        self.c2v_lik = {i: {j:0 for j in (h[h!=-1.])} for i,h in enumerate(H1) }
        
        # VARIABLE2CHECK MESSAGES
        # Create two dictionaries
        # 1. map each variable node index to its neighbouring check node indexes
        # 2. map each variable node index to a dictionary with check node index to variable2check 
        #    message value - initialize variable2check messages as channel likelihoods.
        
        # Transpose parity check matrix for variable2check mapping and multiply each row 
        # with 1-N to fill non-zero cells with an index value subtracting 1 will set 0 value cells 
        # to -1 and allow us to use pythons 0 val indexing.
        H2 = self.H.T*range(1,self.H.shape[0]+1)-1
        # Create variable2check index dict, ignore -1 values as imply no connection
        self.v2c_idx = {i: list(h[h!=-1.]) for i,h in enumerate(H2) }
        # Create variable2check message dict, initialize messages as channel likelihoods
        self.v2c_lik = {i: {j: self.ch_liks[i] for j in (h[h!=-1.])} for i,h in enumerate(H2) }
        
    def _check2variable(self):
        '''
        Update check2variable messages.
        '''
        # Work through each check node
        for c in self.c2v_idx:
            # Get set of neighbor variables for check node
            Nc = set(self.c2v_idx[c])
            # Iterate through each neighbouring variable
            for v in Nc:
                # Nbs excluding current variable node
                Nx = Nc.difference({v})
                # Create array of current messages from neighbor vars to check excluding v
                p = np.array([ self.v2c_lik[n][c] for n in Nx])
                # Apply tanh(./2) to each message
                p = np.tanh(p/2)
                # Take the product of this
                p = np.prod(p)
                # ln( (1 + prod( tanh(m/2) for each m) / (1 - prod( tanh(m/2) for each m )) gives updated message
                l = np.log( (1+p)/(1-p) )
                # Update dict to this new message
                self.c2v_lik[c][v] = l

    def _variable2check(self):
        '''
        Update variable2check messages.
        '''
        # Work through each variable node
        for v in self.v2c_idx:
            # Get set of neighbor checks for variable node
            Mv = set(self.v2c_idx[v])
            # Iterate through each neighbouring check
            for c in Mv:
                # Nbs excluding current check node
                Mx = Mv.difference({c})
                # Update variable to check message
                self.v2c_lik[v][c] = self.ch_liks[v] +  np.sum(np.array([self.c2v_lik[m][v] for m in Mx]))
            # Update variable likelihood
            self.var_liks[v] = self.ch_liks[v] + np.sum(np.array([self.c2v_lik[m][v] for m in Mv]))
            
    def _updatePred(self):
        '''
        Update message prediction from current variable likelihoods. If likelihood +ve then
        more confident that that symbol is a 0, if likelihood -ve then assign value 1.
        '''
        self.xpred = np.array([0. if x>=0. else 1. for x in self.var_liks])
        
    def _checkConvergence(self):
        '''
        Check if our message passing algorithm has converged to a valid codeword.
        If the product of parity check matrix H and prediction x in F2 is 0 then we are done.
        '''
        self.converged = np.sum(np.dot(self.H, self.xpred)%2)==0.

    def message_passing(self):
        '''
        Public function to run message passing algorithm.
        
        Run until either algorithm converges to valid codeword or we exhaust
        number of iterations allowed.
        '''
        while self.l<self.runs and not self.converged:
            # Upate check2variable messages
            self._check2variable()
            # Upate variable2check messages
            self._variable2check()
            # Update prediction
            self._updatePred()
            # Check if converged to valid codeword
            self._checkConvergence()
            # Increment run count
            self.l+=1
        
        # Display outcome of message passing algorithm
        if self.converged:
            print("Successfully converged in %d runs" %self.l)
        else:
            print("Failed to converged in %d runs" %self.runs)

if __name__ == '__main__':
    H1 = np.loadtxt("H1.txt")
    y1 = np.loadtxt("y1.txt")
    eps = 0.1
    decoder = LDPC(H1,y1,eps)

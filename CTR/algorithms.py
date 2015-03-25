# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 23:32:51 2015

@author: patanjali
"""


from csv import DictReader
from math import exp, log, sqrt

dict_profiles = False

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, size, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        print size
        self.n = [0.] * (size)
        self.z = [0.] * (size)
        self.w = [0.] * (size)
    def __str__(self):
        return 'alpha : %s, beta : %s, L1 : %s, L2 : %s, D : %s' %(self.alpha,self.beta,self.L1,self.L2,self.D)
    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x, indices):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = self.w

        # wTx is the inner product of w and x
        wTx = 0.
        for i, x_i in zip(indices,x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]*x_i

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, indices,p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # update z and n
        for i, x_i in zip(indices,x):
            # gradient under logloss
            g = (p - y)*x_i
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

class data:
    def __init__(self,path,D,config,indicators,profiles,enquiries,numeric,interactions,K,
                 normalize_profiles, normalize_enquiries):
        self.file = DictReader(open(path))
        self.D = D
        self.indicators = indicators
        self.profiles = profiles
        self.enquiries = enquiries
        self.numeric = numeric
        self.interactions = interactions
        self.K = K
        self.normalize_profiles = normalize_profiles
        self.normalize_enquiries = normalize_enquiries
        if indicators:
            self.indicator_keys = config.indicator_keys
        else:
            self.indicator_keys = []
        if profiles:
            self.profile_keys = config.profile_keys
        else:
            self.profile_keys = []
        if enquiries:
            self.enquiry_keys = config.enquiry_keys
        else:
            self.enquiry_keys = []
        if numeric:
            self.numeric_keys = config.numeric_keys
        else:
            self.numeric_keys = []
        if interactions:
            self.interaction_keys = config.interaction_keys
        else:
            self.interaction_keys = []
        
        if dict_profiles:
            self._profiles = {}
        else:
            self._profiles = [0]*(D*profiles)
        
        self.profiles_all = [1]*(2)
        self._enquiries = [0]*(D*enquiries)
        self.enquiries_all = 1
        
        self.x = [1.]*(len(self.indicator_keys)+len(self.profile_keys)+len(self.enquiry_keys)+len(self.numeric_keys)+D*self.interactions+1)
        self.indices = [-1]*(len(self.indicator_keys)+len(self.profile_keys)+len(self.enquiry_keys)+len(self.numeric_keys)+D*self.interactions+1)
        self.indices_click = [0]*len(self.profile_keys)
        self.indices_nonclick = [0]*len(self.profile_keys)
        self.indices_today_click = [0]*len(self.profile_keys)
        self.indices_today_nonclick = [0]*len(self.profile_keys)
        self.indices_enq = [0]*len(self.enquiry_keys)
        
        self.size = D*indicators+len(self.profile_keys)+len(self.enquiry_keys)+len(self.numeric_keys)+D*self.interactions+1
        #[x for x in self.file.fieldnames]
        #self.profile_keys.remove('id')
        #self.profile_keys.remove('click')
    def openFile(self,path):    
        self.file = DictReader(open(path))
        
    def items(self):
        ''' GENERATOR: Apply hash-trick to the original csv row
                       and for simplicity, we one-hot-encode everything
    
            INPUT:
                path: path to training or testing file
                D: the max index that we can hash to
    
            YIELDS:
                ID: id of the instance, mainly useless
                x: a list of hashed and one-hot-encoded 'indices'
                   we only need the index since all values are either 0 or 1
                y: y = 1 if we have a click, else we have y = 0
        '''
        D = self.D
        for t, row in enumerate(self.file):
            #if self.subsample < random.random():
            #    continue
            # process id
            ID = row['id']
            del row['id']
    
            # process clicks
            y = 0.
            if 'click' in row:
                if row['click'] == '1':
                    y = 1.
                del row['click']
    
            # extract date
            date = int(row['hour'][4:6])
    
            # turn hour really into hour, it was originally YYMMDDHH
            row['hour'] = row['hour'][6:]
            
            # Indicator variables
            for i, key in enumerate(self.indicator_keys):
                value = row[key]
                index = abs(hash(key + '_' + value)) % D
                self.indices[i] = index
            i = len(self.indicator_keys)
            clicks_all = self.profiles_all[0]
            nonclicks_all = self.profiles_all[1]
            click_pct = (clicks_all*1.0)/(clicks_all+nonclicks_all)
            
            # Profile variables
            for j,key in enumerate(self.profile_keys):                

                index = D*self.indicators + j
                if dict_profiles:
                    index_click = key + '_' + row[key] + '_clicks'
                    index_nonclick = key + '_' + row[key] + '_nonclicks'
                    index_today_click = key + '_' + row[key] + '_clicks_' + str(date)
                    index_today_nonclick = key + '_' + row[key] + '_nonclicks_' + str(date)
                    for temp in [index_click, index_nonclick, index_today_click, index_today_nonclick]:
                        if temp not in self._profiles:
                            self._profiles[temp] = 0
                else:
                    index_click = abs(hash(key + '_' + row[key] + '_clicks')) % D
                    index_nonclick = abs(hash(key + '_' + row[key] + '_nonclicks')) % D
                    index_today_click = abs(hash(key + '_' + row[key] + '_clicks_' + str(date))) % D
                    index_today_nonclick = abs(hash(key + '_' + row[key] + '_nonclicks_' + str(date))) % D
                
                num = (self._profiles[index_click] - self._profiles[index_today_click] + self.K*click_pct)
                den = (self._profiles[index_nonclick]+self._profiles[index_click]-\
                        self._profiles[index_today_nonclick] - self._profiles[index_today_click]+self.K)
                
                if self.normalize_profiles:
                    self.x[i+j] = num/den/click_pct
                else:
                    self.x[i+j] = num/den
                #print key, row[key], x[-1], y
                self.indices[i+j] = index
                self.indices_click[j] = index_click
                self.indices_nonclick[j] = index_nonclick
                self.indices_today_click[j] = index_today_click
                self.indices_today_nonclick[j] = index_today_nonclick
            j = len(self.profile_keys)
            
            # Enquiry variables
            for l,key in enumerate(self.enquiry_keys):
                index = D*self.indicators + j + l
                index_inq = abs(hash(key+'_'+row[key]+'_views')) % D
                if self.normalize_enquiries:
                    self.x[i+j+l] = self._enquiries[index_inq]/(self.enquiries_all*1.0)
                else:
                    self.x[i+j+l] = self._enquiries[index_inq]
                self.indices[i+j+l] = index
                self.indices_enq[l] = index_inq
                self._enquiries[index_inq] += 1
            self.enquiries_all += 1
            l = len(self.enquiry_keys)
            
            # Numeric variables
            for m, key in enumerate(self.numeric_keys):
                index = D*self.indicators + j + l + m
                self.x[i+j+l+m] = float(row[key])
                self.indices[i+j+l+m] = index
            m = len(self.numeric_keys)
            # Interaction variables
            for n, keys in enumerate(self.interaction_keys):
                value1 = row[keys[0]]
                value2 = row[keys[1]]
                index = D*self.indicators + j + l + m + abs(hash(keys[0] + '_' + value1 + '_' + keys[1] + '_' + value2)) % D
                self.indices[i+j+l+m+n] = index
            
            yield t, date, ID, self.x, y, self.indices, \
                    self.indices_click, self.indices_nonclick, \
                    self.indices_today_click, self.indices_today_nonclick, \
                    self.indices_enq
    
    def update_profiles(self,y,indices_click,indices_nonclick,
                        indices_today_click,indices_today_nonclick,
                        indices_enquiries):
        """
        # Update enquiries - Moving this to lookup made more sense
        if self.enquiries:
            self.enquiries_all += 1
            for index in indices_enquiries:
                self._enquiries[index] += 1
        """
        # Update profiles
        if y == 1:
            self.profiles_all[0] += 1
            for index in indices_click:
                self._profiles[index] += 1
            for index in indices_today_click:
                self._profiles[index] += 1
        else:
            self.profiles_all[1] += 1
            for index in indices_nonclick:
                self._profiles[index] += 1
            for index in indices_today_nonclick:
                self._profiles[index] += 1

def dict_to_str(dictionary):
    return '_'.join([str(key) + '_' + str(dictionary[key]) for key in dictionary])
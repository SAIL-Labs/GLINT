import numpy as np

class Chip():
    def __init__(self, input_wave, n_apertures = 3, n_phot = 3, n_tricoupler = 3, n_directional = 0, coeff_photo = np.array([1/3, 1/3, 1/3]), coeff_interf = np.array([1/2, 1/2, 1/2])):
        self.input_wave = input_wave
        self.n_apertures = n_apertures
        self.n_phot = n_phot
        self.alpha = coeff_photo 
        self.beta = coeff_interf  
        self.n_tricoupler = n_tricoupler
        self.n_directional = n_directional

        self.check_parameters()

    def check_parameters(self):
        '''
        Just some basic checks to make sure the parameters are valid.
        '''

        if self.n_apertures < 1:
            raise ValueError("Number of apertures must be at least 1")
        if self.n_phot < 0:
            raise ValueError("Number of photometry channels must be at least 0")
        if self.n_tricoupler < 0:
            raise ValueError("Number of tricouplers must be at least 0")
        if self.n_directional < 0:
            raise ValueError("Number of directional couplers must be at least 0")
        if len(self.alpha) != self.n_phot:
            raise ValueError("Number of photometry coefficients must match number of photometry channels")
        if len(self.beta) != self.n_apertures:
            raise ValueError("Number of interferometry coefficients must match number of apertures")
        if len(self.input_wave) != self.n_apertures:
            raise ValueError("Number of input waves must match number of apertures")
        if self.n_phot > self.n_apertures:
            raise ValueError("Number of photometry channels must be less than or equal to number of apertures")
    
    def splitter_photometry(self, theta = 0):
        '''
        Photometry splitter matrix.
        '''

        interferometry = np.sqrt(1-self.alpha) * np.eye(self.n_apertures, dtype = complex) 
        photometry = np.sqrt(self.alpha) * np.eye(self.n_apertures, dtype = complex)[0:self.n_phot] # limit the photometry channels to the n_photo specified

        # In theory there is some phase delay introduced by the splitter, but there is no phase difference between the two outputs of the y-junction
        S = np.exp(1j*theta) * np.vstack((interferometry, photometry)) 

        return S

    def splitter_interferometry(self, theta = 0):
        '''
        This function creates the splitter for the interferometry channels and remaps the matrix.
        '''
        
        # Vector for one splitter
        splitter = np.array([np.sqrt(self.beta), np.sqrt(1-self.beta)])

        # Resizing the vector such that there is a splitter for each input and photometry channels. 
        # E.g. for 2 inputs and 2 photometry channels, the shape would need to take in 4 inputs (2 inputs and 2 photometry channels)
        # and have 6 outputs (4 outputs from splitters and 2 photometry channels)
        splitter_resized = np.zeros((2*self.n_apertures + self.n_phot, self.n_apertures + self.n_phot), dtype = complex)

        # Insert the splitter vector into the resized vector
        splitter_resized[:splitter.shape[0], :splitter.shape[1]] = splitter
        splitter_resized_T = splitter_resized.T

        # Shift the splitter vector such that the first splitter is for the first input, the second splitter is for the second input, etc.
        # i.e. shift column 1 down two spots and column 3 down 4 spots etc.
        for i in range(self.n_apertures):
            splitter_resized_T[i] = np.roll(splitter_resized_T[i], i*2)

        S = splitter_resized_T.T

        # Add the phase term
        S = np.exp(1j*theta)*S

        # Add the photometry channels.
        # E.g. for 2 inputs and 2 photometry channels, the vector would be (excluding phase term):
        # [  sqrt(beta),            0, 0, 0]
        # [sqrt(1-beta),            0, 0, 0]
        # [           0,   sqrt(beta), 0, 0]
        # [           0, sqrt(1-beta), 0, 0]
        # [           0,            0, 1, 0]
        # [           0,            0, 0, 1]
        S[-1*self.n_phot:, -1*self.n_phot:] = np.eye(self.n_phot)

        # Remapping: e.g. for 3 inputs, we have the following pairs of input beam combinations (1,2), (2,3), (3,1)
        # Before remapping, we have two beams for each aperture/input i.e. (1,1*), (2,2*), (3,3*)
        # To remap, we SWAP the inner pairs i.e. (1,2), (1*,3), (2*,3*)
        start_row = 1
        n_couplers = self.n_tricoupler + self.n_directional
        end_row = 2*n_couplers-1  # Finish swapping the pairs before we get to the photometry channels

        for i in range(start_row, end_row, 2):
            S[[i, i+1], :] = S[[i+1, i], :]


        return S
    

    def phase_shifter(self, theta = np.pi):

        '''
        Phase shifter matrix.
        '''

        n_inputs = 2*(self.n_tricoupler+self.n_directional) # number of wg after interferometry
        n_outputs = n_inputs + self.n_phot

        P = np.eye(n_outputs, dtype = complex)

        # Starting from the second row of P
        for i in range(1, n_inputs, 2):
            # Make every second diagonal element equal to np.exp(1j*np.pi) excluding photometry channels
            P[i, i] = np.exp(1j*theta)

        # Now add a row of zeros between each row of 1 and exp(1j*theta) which act as the third waveguide for each tricoupler.
        P = np.insert(P, np.arange(1, self.n_tricoupler*2, 2), 0, axis = 0)

        # E.g. For 2 inputs with 2 tricouplers and 2 photometry channels, the matrix would be:
        # [1,                0, 0,                0, 0, 0]
        # [0,                0, 0,                0, 0, 0]
        # [0, np.exp(1j*theta), 0,                0, 0, 0]
        # [0,                0, 1,                0, 0, 0]
        # [0,                0, 0,                0, 0, 0]
        # [0,                0, 0, np.exp(1j*theta), 0, 0]
        # [0,                0, 0,                0, 1, 0]
        # [0,                0, 0,                0, 0, 1]

        return P


    def couplers(self):
        '''
        This function creates the coupler matrix. If a mixture of tricouplers and directional couplers are used, it places the tricouplers in the top half of the matrix.
        '''

        # Find how many coupling waveguides there are 
        size_dir = self.n_directional*2
        size_tri = self.n_tricoupler*3
        size_coupler = size_dir + size_tri

        # Tricoupler matrix
        Atri = np.exp(1j*2*np.pi/3)*np.ones((3,3))
        np.fill_diagonal(Atri, 1)
        Atri = Atri*np.sqrt(1/3)

        # Directional matrix
        Adir = np.exp(1j*np.pi/2)*np.ones((2,2))
        np.fill_diagonal(Adir, 1)
        Adir = Adir*np.sqrt(1/2)

        # Size up the tricoupler matrix to match the number of tricouplers
        Btri = np.kron(np.eye(self.n_tricoupler), Atri)
        
        # Size up the directional matrix to match the number of directional couplers
        Bdir = np.kron(np.eye(self.n_directional), Adir) if self.n_directional > 0 else np.array([])

        # Combine the matrices
        size_final = size_coupler + self.n_phot
        final_matrix = np.zeros((size_final, size_final), dtype=complex)
        final_matrix[:size_tri, :size_tri] = Btri
        final_matrix[size_tri:size_tri+size_dir, size_tri:size_tri+size_dir] = Bdir
        final_matrix[size_coupler:, size_coupler:] = np.eye(self.n_phot)

        return final_matrix
    
    def assemble(self):
        '''
        This function assembles the chip by multiplying the splitter, interferometer, phase shifter and coupler matrices.
        '''
        
        Sp = self.splitter_photometry()
        Si = self.splitter_interferometry()
        P = self.phase_shifter(theta = np.pi)
        C = self.couplers()

        output = np.dot(C, np.dot(P, np.dot(Si, np.dot(Sp, self.input_wave))))

        return output
    







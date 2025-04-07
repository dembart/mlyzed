import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import matplotlib.pyplot as plt

class Arrhenius:

    def __init__(self, temperatures, msd_list):
        self.temperatures = np.array(temperatures)
        self.d = np.array([msd.diffusivity[0] for msd in msd_list])
        self.d_err = np.array([msd.diffusivity[1] for msd in msd_list])
        self.msd_list = msd_list
        self._fit()


    def _fit(self):
        
        temp = np.array(self.temperatures)
        d = self.d
        d_err = self.d_err
        
        def line(x, intercept, slope):
            y = slope * x + intercept
            return y
        
        def exponent(x, intercept, slope):
            return intercept * np.exp(-slope/x)
        
        #x = 1/temp
        #y = np.log(d)
        #yerr = d_err/d
        x = temp
        y = d
        yerr = d_err
        popt, pcov = curve_fit(exponent, x, y, sigma = yerr, absolute_sigma = True)
        intercept, slope = popt
        intercept_err, slope_err = np.sqrt(np.diag(pcov))
        #intercept_err = np.sqrt(np.diag(pcov))[0]

        residuals = y - line(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        self.slope = slope 
        self.slope_err = slope_err 
        self.intercept = intercept
        self.intercept_err = intercept_err
        self.r_squared = r_squared
        self.barrier = self.slope * 8.617e-5
        self.barrier_err = self.slope_err * 8.617e-5
        self.prefacor = self.intercept
        #d(lnx) = dx/x -> dx = x * d(lnx), x = D0, d(lnx) = self.intercept_err
        self.prefacor_err =self.intercept_err
        #self.intercept_err * np.exp(self.intercept)


    def predict_diffusivity(self, T):

        d = np.exp(-self.slope * 1/T + self.intercept)
        d_lower = np.exp((self.slope + self.slope_err) * 1/T + self.intercept - self.intercept_err)
        d_upper = np.exp((self.slope - self.slope_err) * 1/T + self.intercept + self.intercept_err)
        return d, d_lower, d_upper
    
    

    def equation(self):
        text = f'Fit (R^2 = {np.round(self.r_squared, 3)}), D = {self.factor} exp (-{self.barrier} eV / kT) cm^2/s'
        return text


    def plot(self, show = True, axes = None, dpi = 150, figsize = (7, 3)):

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(dpi = dpi, figsize = figsize, ncols = 2)
        else:
            fig = plt.gcf()
            ax1, ax2 = axes
        
        for (t, msd) in zip(self.temperatures, self.msd_list):
            ax1.plot(msd.dt, msd.msd, label = f'{t} K')
            if np.any(msd.msd_std):
                ax1.fill_between(msd.dt, msd.msd - msd.msd_std, msd.msd + msd.msd_std,
                                    alpha = 0.3)
            slope, intercept, slope_err, intercept_err, r_squared = msd.fit_line()
            ax1.plot(msd.dt, msd.dt * slope + intercept, color = 'k', linewidth = 0.75, linestyle = '--')
        ax1.set_xlabel('lagtime, ps')
        ax1.set_ylabel('MSD, $\AA^{2}$')
        ax1.grid(alpha = 0.5)
        ax1.set_ylim(0, max([msd.msd.max() for msd in self.msd_list]))
        ax1.set_xlim(min([msd.dt.min() for msd in self.msd_list]),
                     max([msd.dt.max() for msd in self.msd_list]))
        ax1.legend()

        x = 1000/self.temperatures
        y = np.log10(self.d)
        yerr = self.d_err / (np.log(10) * self.d)
        ax2.errorbar(x, y, yerr = yerr, linestyle = '',
                        capsize=2,
                        color = 'darkred',
                        markeredgewidth = 0.75, 
                        linewidth = 0.75,
                        marker = 'o',
                        markersize = 4,
                        label = 'data',
                    )
        
        ax2.set_xlabel('1000/T, 1/K')
        ax2.set_ylabel('log$_{10}$$D$, $cm^{2}/s$')


        ax2.plot(x, (1e-3/np.log(10) * (-self.slope) * x + np.log10(self.intercept)), color = 'k',
                 label = 'fit')
        ax2.legend()
        ax2.grid(alpha = 0.5, linewidth = 0.5)
        plt.tight_layout()

        if show:
            plt.show()
        return fig, (ax1, ax2)

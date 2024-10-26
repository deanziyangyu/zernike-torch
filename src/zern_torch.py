#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2024 Ziyang Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Complex- and real-valued Zernike polynomials.
Adapted from Zernike by Jacopo Antonello et al.
Originally https://github.com/jacopoantonello/zernike
"""

from abc import ABC, abstractmethod
from math import factorial

import numpy as np
import torch
from torch.linalg import lstsq

class Zern(ABC):
    """Shared code for `RZern` and `CZern`.

    This is an abstract class, use `RZern` and `CZern` instead. Only
    `NORM_NOLL` is implemented. The polynomials are ordered and normalised
    according to [N1976]_ and [M1994]_, see also Appendix A in [A2015]_.

    References
    ----------
    ..  [N1976] R. Noll, "Zernike polynomials and atmospheric turbulence,"
        J. Opt. Soc. Am.  66, 207-211 (1976). `doi
        <http://dx.doi.org/10.1364/JOSA.66.000207>`__.
    ..  [M1994] V. N. Mahajan, "Zernike circle polynomials and optical
        aberrations of systems with circular pupils," Appl. Opt. 33, 8121-8124
        (1994). `doi <http://dx.doi.org/10.1364/AO.33.008121>`__.
    ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
        retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
        (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

    """

    # unimplemented, ENZ papers of Janssen, Braat, etc.
    NORM_ZERNIKE = 0
    # Noll/Mahajan's normalisation, unit variance over the unit disk
    NORM_NOLL = 1

    def _print_rhotab(self):
        for i in range(self.nk):
            for j in range(self.n + 1):
                print('{:< 3.3f} '.format(self.rhotab[i, j]), end='')
            print('')

    def _print_nmtab(self):
        for i in range(self.nk):
            print('{:< d} {:< d}'.format(self.ntab[i], self.mtab[i]))

    def _make_rhotab_row(self, c, n, m):
        # col major, row i, col j
        self.coefnorm[c] = self.ck(n, m)
        for s in range((n - m) // 2 + 1):
            self.rhotab[c, self.n - (n - 2 * s)] = (
                ((-1)**s) * factorial(n - s) /
                (factorial(s) * factorial((n + m) // 2 - s) *
                 factorial((n - m) // 2 - s)))

    def __init__(self, n, normalise=NORM_NOLL, device=None):
        """Initialise Zernike polynomials up to radial order `n`.

        This is an abstract class, use `RZern` and `CZern` instead. Only
        `NORM_NOLL` is implemented.

        Args:
            n (int): radial order.
            normalise (int): normalisation scheme.
            device (str): device to use, e.g., 'cpu' or 'cuda'.

        """

        self.shape = None
        self.torch_dtype = 'undefined'
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        nk = (n + 1) * (n + 2) // 2
        self.n = n
        self.nk = nk
        self.normalise = normalise
        assert (self.normalise == self.NORM_NOLL)

        # coefficients of R_n^m(\rho), see [N1976]_
        rhotab = torch.zeros((nk, n + 1), dtype=torch.float32, device=self.device)
        # coefficients of \int R_n^m(\rho) \rho \,d\rho
        rhoitab = torch.zeros((nk, n + 3), dtype=torch.float32, device=self.device)
        ntab = torch.zeros(nk, dtype=torch.int32, device=self.device)
        mtab = torch.zeros(nk, dtype=torch.int32, device=self.device)
        coefnorm = torch.zeros(nk, dtype=torch.float32, device=self.device)

        self.rhotab = rhotab
        self.rhoitab = rhoitab
        self.ntab = ntab
        self.mtab = mtab
        self.coefnorm = coefnorm

        self.rhotab[0, n] = 1.0
        self.coefnorm[0] = 1.0

        for ni in range(1, n + 1):
            for mi in range(-ni, ni + 1, 2):
                k = self.nm2noll(ni, mi) - 1
                self._make_rhotab_row(k, ni, abs(mi))
                ntab[k], mtab[k] = ni, mi

        # make rhoitab
        for ci in range(nk):
            for ni in range(n + 1):
                self.rhoitab[ci, ni] = self.rhotab[ci, ni] / (n + 2 - ni)

    @staticmethod
    def nm2noll(n, m):
        """Convert indices `(n, m)` to the Noll's index `k`.

        Note that Noll's index `k` starts from one and Python indexing is
        zero-based.

        """
        k = n * (n + 1) // 2 + abs(m)
        if (m <= 0 and n % 4 in (0, 1)) or (m >= 0 and n % 4 in (2, 3)):
            k += 1
        return k


    def Rnm(self, k, rho):
        r"""Compute the `k`-th radial polynomial :math:`R_n^m(\rho)`.

        The radial polynomial is defined in Eq. (2) of [N1976]_ and Eq. (2) of
        [M1994]_.

        References
        ----------
        ..  [N1976] R. Noll, "Zernike polynomials and atmospheric turbulence,"
            J. Opt. Soc. Am.  66, 207-211 (1976). `doi
            <http://dx.doi.org/10.1364/JOSA.66.000207>`__.
        ..  [M1994] V. N. Mahajan, "Zernike circle polynomials and optical
            aberrations of systems with circular pupils," Appl. Opt.  33,
            8121–8124 (1994). `doi
            <http://dx.doi.org/10.1364/AO.33.008121>`__.

        """
        return self.polyval(self.rhotab[k, :], rho)


    def polyval(self, p, x):
        """Adapted for `numpy.polyval`.
        poly1d function is unimplemented
        """
        y = torch.zeros_like(x, device=self.device)
        for i in range(len(p)):
            y = y * x + p[i]
        return y


    def I_Rnmrho(self, k, rho):
        r"""Compute :math:`\int R_n^m(\rho)\rho`."""
        return sum([(rho**(self.n + 2 - i)) * self.rhoitab[k, i]
                    for i in range(self.n + 3)])

    @abstractmethod
    def angular(self, k, theta):
        pass

    def radial(self, k, rho):
        return self.coefnorm[k] * self.Rnm(k, rho)

    def vect(self, Phi):
        r"Reshape `Phi` into a vector"
        return Phi.flatten()

    def matrix(self, Phi):
        r"Reshape `Phi` into a matrix"
        if self.shape is None:
            raise ValueError('Use make_cart_grid() to define the shape first')
        elif self.shape[0] * self.shape[1] != Phi.numel():
            raise ValueError('Phi.shape should be {}'.format(self.shape))
        return Phi.view(self.shape)

    def make_cart_grid(self, xx, yy, unit_circle=True):
        r"""Make a cartesian grid to evaluate the Zernike polynomials.

        Parameters
        ----------
        - `xx`: `torch` tensor generated with `torch.meshgrid()`.
        - `yy`: `torch` tensor generated with `torch.meshgrid()`.
        - `unit_circle`: set `np.nan` for points where :math:`\rho > 1`.

        Examples
        --------

        .. code:: python

            import torch
            from zernike import RZern

            cart = RZern(6)
            dd = torch.linspace(-1.0, 1.0, 200)
            xv, yv = torch.meshgrid(dd, dd)
            cart.make_cart_grid(xv, yv)

        Notes
        -----
        `ZZ` is stored with `order='F'`.

        """
        self.ZZ = torch.zeros((xx.numel(), self.nk),
                           dtype=self.torch_dtype,
                           device=self.device)
        self.shape = xx.shape
        rho = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
        for k in range(self.nk):
            prod = self.radial(k, rho) * self.angular(k, theta)
            if unit_circle:
                prod[rho > 1.0] = float('nan')
            self.ZZ[:, k] = self.vect(prod)

    def eval_grid(self, a, matrix=False):
        """Evaluate the Zernike polynomials using the coefficients in `a`.

        Parameters
        ----------
        - `a`: Zernike coefficients.

        Returns
        -------
        -   `Phi`: a `torch` vector in column major order. Use the `vect()` or
            `matrix()` methods to flatten or unflatten the matrix.
        -   `matrix`: return a matrix instead of a vector

        Examples
        --------

        .. code:: python

            import torch
            import matplotlib.pyplot as plt
            from zernike import RZern

            cart = RZern(6)
            L, K = 200, 250
            ddx = torch.linspace(-1.0, 1.0, K)
            ddy = torch.linspace(-1.0, 1.0, L)
            xv, yv = torch.meshgrid(ddx, ddy)
            cart.make_cart_grid(xv, yv)

            c = torch.zeros(cart.nk)
            plt.figure(1)
            for i in range(1, 10):
                plt.subplot(3, 3, i)
                c *= 0.0
                c[i] = 1.0
                Phi = cart.eval_grid(c, matrix=True)
                plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
                plt.axis('off')

            plt.show()

        """
        if a.numel() != self.nk:
            raise ValueError('a.size = {} but self.nk = {}'.format(
                a.numel(), self.nk))
        Phi = torch.matmul(self.ZZ, a)
        if matrix:
            return self.matrix(Phi)
        else:
            return Phi

    def fit_cart_grid(self, Phi, rcond=None):
        """Fit a cartesian grid using least-squares.

        Parameters
        ----------
        - `Phi`: cartesian grid, e.g., generated with make_cart_grid().
        - `rcond`: rcond supplied to `lstsq`

        Returns
        -------
        -   `a`, `torch` vector of Zernike coefficients
        -   `res`, see `lstsq`
        -   `rnk`, see `lstsq`
        -   `sv`, see `lstsq`

        Examples
        --------

        .. code:: python

            import torch
            import matplotlib.pyplot as plt
            from zernike import RZern

            cart = RZern(6)
            L, K = 200, 250
            ddx = torch.linspace(-1.0, 1.0, K)
            ddy = torch.linspace(-1.0, 1.0, L)
            xv, yv = torch.meshgrid(ddx, ddy)
            cart.make_cart_grid(xv, yv)

            c0 = torch.randn(cart.nk)
            Phi = cart.eval_grid(c0, matrix=True)
            c1 = cart.fit_cart_grid(Phi)[0]
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.plot(range(1, cart.nk + 1), c0, marker='.')
            plt.plot(range(1, cart.nk + 1), c1, marker='.')

            plt.show()

        """
        vPhi = self.vect(Phi)
        zfm = torch.logical_and(torch.isfinite(self.ZZ[:, 0]), torch.isfinite(vPhi))
        zfA = self.ZZ[zfm, :]
        Phi1 = vPhi[zfm]

        a, res, rnk, sv = lstsq(torch.matmul(zfA.T, zfA),
                                torch.matmul(zfA.T, Phi1),
                                rcond=rcond)

        return a, res, rnk, sv

class CZern(Zern):
    r"""Complex-valued Zernike polynomials.

    .. math::

        \mathcal{N}_k(\rho, \theta) = \mathcal{N}_n^m(\rho, \theta)

        \mathcal{N}_n^m(\rho, \theta) = c_n^m R_n^{|m|}(\rho)\exp(i m\theta)

        c_n^m = \sqrt{n + 1}

        \int |mathcal{N}_k(rho, theta)|^2 \rho d\rho d\theta = \pi, \;
        \text{for} \; k > 1

    See Eq. (A5) in [A2015]_.

    References
    ----------
    ..  [N1976] R. Noll, "Zernike polynomials and atmospheric turbulence,"
        J. Opt. Soc. Am.  66, 207-211 (1976). `doi
        <http://dx.doi.org/10.1364/JOSA.66.000207>`__.
    ..  [M1994] V. N. Mahajan, "Zernike circle polynomials and optical
        aberrations of systems with circular pupils," Appl. Opt. 33, 8121–8124
        (1994). `doi <http://dx.doi.org/10.1364/AO.33.008121>`__.
    ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
        retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
        (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

    """
    def __init__(self, n, normalise=Zern.NORM_NOLL, device=None):
        super(CZern, self).__init__(n, normalise, device)
        self.torch_dtype = torch.complex64

    def ck(self, n, m):
        return torch.sqrt(torch.tensor(n + 1.0, dtype=torch.float32, device=self.device))

    def angular(self, j, theta):
        m = self.mtab[j]
        return torch.exp(1j * m * theta)


class RZern(Zern):
    r"""Real-valued Zernike polynomials.

    .. math::

        \mathcal{Z}_k(\rho, \theta) = \mathcal{Z}_n^m(\rho, \theta)

        \mathcal{Z}_n^m(\rho, \theta) = c_n^m R_n^{|m|}(\rho)
        \Theta_n^m(\theta)

        c_n^m =
        \begin{cases}
            \sqrt{n + 1} & m = 0\\
            \sqrt{2(n + 1)} & m \neq 0
        \end{cases}

        \Theta_n^m(\theta) =
        \begin{cases}
            \cos(m\theta) & m \ge 0\\
            -\sin(m\theta) & m < 0
        \end{cases}

        \int |\mathcal{Z}_k(rho, theta)|^2 \rho d\rho d\theta = \pi, \;
        \text{for} \; k > 1

    See Eq. (A1) in [A2015]_.

    References
    ----------
    ..  [N1976] R. Noll, "Zernike polynomials and atmospheric turbulence,"
        J. Opt. Soc. Am.  66, 207-211 (1976). `doi
        <http://dx.doi.org/10.1364/JOSA.66.000207>`__.
    ..  [M1994] V. N. Mahajan, "Zernike circle polynomials and optical
        aberrations of systems with circular pupils," Appl. Opt. 33, 8121–8124
        (1994). `doi <http://dx.doi.org/10.1364/AO.33.008121>`__.
    ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
        retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
        (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

    """
    def __init__(self, n, normalise=Zern.NORM_NOLL, device=None):
        super(RZern, self).__init__(n, normalise, device)
        self.torch_dtype = torch.float32

    def ck(self, n, m):
        if self.normalise == self.NORM_NOLL:
            if m == 0:
                return torch.sqrt(torch.tensor(n + 1.0, dtype=torch.float32, device=self.device))
            else:
                return torch.sqrt(torch.tensor(2.0 * (n + 1.0), dtype=torch.float32, device=self.device))
        else:
            return torch.tensor(1.0, dtype=torch.float32, device=self.device)

    def angular(self, j, theta):
        m = self.mtab[j]
        if m >= 0:
            return torch.cos(m * theta)
        else:
            return torch.sin(-m * theta)

"""
This module provides an implementation of the Rouse model.

Its content is mostly the `Model` class, so see there for more description.
Otherwise, there is the functional form for the MSD of two loci on an infinite
continuous Rouse polymer in `twoLocusMSD`. This might be used as
phenomenological expression for fitting to data.

See also
--------
Model
"""
import numpy as np
import scipy.linalg
import scipy.special

class Model:
    r"""
    The Rouse model

    This model follows the linear Langevin equation

    .. math:: \dot{x}(t) = -kAx(t) + F + \xi(t)\,,

    where :math:`x` is a vector of monomer positions and :math:`\xi` is
    Gaussian white noise of strength proportional to `D` driving the system.
    The connectivity matrix :math:`A` describes which monomers are connected to
    which by harmonic bonds with spring constant :math:`k`. The connectivity
    matrix is conveniently set up using `setup_free_chain` and
    `add_bonds`.  Finally, :math:`F` can be used to incorporate an
    external force on individual monomers.

    This model implementation provides the following functionality:

    - setting up a spring network with arbitrary connectivity matrix
    - calculating the steady state ensemble.
    - propagating an arbitrary Gaussian ensemble forward in time
    - evolving a single conformation :math:`x` forward in time
    - analytically calculating MSD & ACF, also for arbitrary linear
      combinations of loci (e.g. MSD of the vector between two loci)
    - other auxiliary stuff, like giving equilibration time scales and contact
      probabilities

    Attributes
    ----------
    N : int
        number of monomers
    D : float
        diffusivity of an individual monomer (i.e. strength of the driving
        white noise)
    k : float
        default spring constant for harmonic bonds
    d : int
        spatial dimension
    A : (N, N) array, dtype=float, internal
        connectivity matrix. Initialized to a linear polymer by default
    F : (N, d) array, dtype=float, internal
        deterministic external force on each monomer. Zero by default

    Other Parameters
    ----------------
    setup_dynamics : bool
        whether to pre-calculate matrices needed for dynamics (propagation of
        ensembles / evolution of single conformation).
    add_bonds : list of bonds
        shortcut for adding bonds to the system before initializing dynamics.
        Corresponds to the `!links` argument of `add_bonds`, see there for
        more details.
    """
    # Implementation notes
    # --------------------
    # In a decision to favor time over space, we use simple numpy arrays
    # instead of sparse matrices (which might otherwise make sense for A).
    #
    # The check_dynamics mechanism is supposed to simply catch the most
    # straight-forward mistakes, i.e. it checks the parameter values and keeps track
    # of changes to the matrices through the member functions, but nothing else.
    #
    # This implementation uses S = 1 and AS = SA^T, i.e. we also assume symmetric A.
    #
    # D can be zero
    #
    # Numerics
    # --------
    # Since A is singular in many use cases, getting the numerics straight
    # requires some thought. Also some conceptual points such as "what's the
    # right steady state?" might become somewhat tricky.
    # 
    # The numerical issue is actually not limiting in our case: A is always
    # real & symmetric, thus orthogonally diagonalizable, which eliminates
    # numerical issues with the eigendecomposition (Moler & van Loan, 1978: "19
    # dubious ways ..."). This should thus be the method of choice. Thus:
    #
    # - propagation requires M <-- B.M + G, C <-- B.C.B^T + Σ with B =
    #   exp(-k*A*dt). Following the above, we calculate B via EV decomposition.
    # - the "propagation matrices" G and Σ require integration of exp(A).
    #   Following the above, we again use EV decomposition. Note that small
    #   eigenvalues will have to be replaced by hand
    # - for the steady state, we have <x> = (kA)^{-1}.F and <x.x^T>_c =
    #   D/k*A^{-1}. These expressions correctly diverge when A is not
    #   invertible, as there is no steady state. In this case, however, it
    #   makes sense to fix all non-steady state degrees of freedom (i.e. the
    #   center of mass) at 0 and study the steady state for the remaining ones.
    #   This is exactly achieved by the Moore-Penrose pseudo-inverse of A,
    #   which we get from the EV decomposition by setting 1/λ = 0 for all
    #   vanishingly small eigenvalues λ.
    # - for multiplication D.X, where D is diagonal with diagonal vector d and
    #   X is arbitrary, use D.X = d[:, None]*X. Similarly X.D = X*d[None, :]
    #   and consequently Y.D.X = (Y*d[None, :]) @ X
    # - numpy docs recommend using "@" for matrix multiplication wherever
    #   possible (numpy.org/doc/stable/reference/routines.linalg.html)
    # - try to keep matrix dimensions as small as possible when multiplying.
    #   Thus, if we multiply V.D.VT.F with V: (N, N), D: diagonal(d), F: (N, 3)
    #   we should write V @ (d[:, None] * (V @ F)). Thus we never actually
    #   execute an (N, N) x (N, N) multiplication
    
    def __init__(self, N, D=1., k=1., d=3, setup_dynamics=True, add_bonds=None):
        self.N = N
        self.D = D
        self.k = k

        self._dynamics = {'needs_updating' : True}

        self.setup_free_chain(d)
        if add_bonds is not None:
            self.add_bonds(add_bonds)

        if setup_dynamics:
            self.update_dynamics()

    @property
    def d(self):
        return self.F.shape[1]

    def __eq__(self, other):
        for param in ['N', 'D', 'k']:
            if getattr(self, param) != getattr(other, param):
                return False
        for param in ['A', 'F']:
            if np.any(getattr(self, param) != getattr(other, param)):
                return False
        return True

    def __repr__(self):
        n_extra_bonds = np.count_nonzero(np.triu(self.A, k=2).flatten())
        rep = "rouse.Model(N={}, D={}, k={}, d={})".format(self.N, self.D, self.k, self.d)
        if n_extra_bonds > 0:
            rep += " with {} additional bonds".format(n_extra_bonds)
        return rep

####### Setting up a model and its dynamics (mostly for internal use)

    def setup_free_chain(self, d=3):
        """
        Set up connectivity matrix of a linear chain and zero external force

        Mostly used internally upon initialization

        Parameters
        ----------
        d : int
            spatial dimension

        See also
        --------
        add_bonds, add_tether
        """
        self._dynamics['needs_updating'] = True

        if self.N == 1:
            self.A = np.zeros((1, 1), dtype=float)
        else:
            self.A = np.diagflat(self.N*[2.], k=0) \
                     + np.diagflat((self.N-1)*[-1.], k=-1) \
                     + np.diagflat((self.N-1)*[-1.], k= 1)
            self.A[0, 0] = self.A[-1, -1] = 1
        
        self.F = np.zeros((self.N, d))

    def add_bonds(self, links, k_rel=1.):
        """
        Add additional bonds to connectivity matrix

        Parameters
        ----------
        links : list of (monA, monB, k_rel) or (monA, monB) tuples
            bonds to add. Introduces an additional bond between ``monA`` and
            ``monB`` (integer indices < `N`) of strength ``k_rel*k``. Giving
            ``k_rel`` as third entry in the tuple overwrites use of the
            function argument `!k_rel` for this bond.
        k_rel : float > 0, optional
            strength of the new bonds, as multiple of the class attribute `!k`.

        See also
        --------
        add_tether
        """
        self._dynamics['needs_updating'] = True

        for link in links:
            myk_rel = k_rel if len(link) == 2 else link[2]
            self.A[link[0], link[0]] += myk_rel
            self.A[link[0], link[1]] -= myk_rel
            self.A[link[1], link[0]] -= myk_rel
            self.A[link[1], link[1]] += myk_rel

    def add_tether(self, mon=0, k_rel=1., point=None):
        """
        Tether a monomer to a fixed point in space

        This simply introduces an additional harmonic bond between this monomer
        and the specified point. Note that you can give a large `!k_rel` to
        actually pin the monomer in place

        Parameters
        ----------
        mon : int
            the index of the monomer that should get the tether
        k_rel : float > 0, optional
            strength of the tether, in multiples of the default bond strength `!k`.
        point : (d,) array-like, optional
            the point to tether to; defaults to the origin

        See also
        --------
        add_bonds
        """
        if point is None:
            point = np.zeros(self.d)
        else:
            point = np.asarray(point)

        self._dynamics['needs_updating'] = True
        
        self.A[mon, mon] += k_rel
        self.F[mon] += self.k * k_rel * point

    def update_dynamics(self, dt=1.):
        """
        Pre-calculate dynamic matrices

        This sets the model up for evaluation of model dynamics, such as
        evolving a conformation or propagating an ensemble. It should be called
        after any modification to model parameters, which is handled
        automatically for internal modifications (such as when `add_bonds`
        is called). If you modify anything externally, recommended behavior is
        to simply set ``model._dynamics['needs_updating'] = True``, which will
        ensure that stuff is recalculated as needed.

        Ultimately, this is just a wrapper for the ``update_*`` functions. If
        you know what you're doing, you can also just call these individually
        as needed.

        Parameters
        ----------
        dt : float
            time step to prepare for

        See also
        --------
        check_dynamics
        """
        # Eigendecomposition of A
        w, V = scipy.linalg.eigh(self.A)
        w[np.abs(w) < 1e-10] = 0

        mp_kw = np.zeros_like(w)
        mp_kw[w != 0] = 1 / (self.k*w[w!=0])

        # Steady state covariance = Moore-Penrose inverse
        ss_CoD = V @ (mp_kw[:, None] * V.T)
        
        # Propagator (simply exp(-k*A*Δt)
        B = V @ (np.exp(-self.k*w*dt)[:, None] * V.T)

        # Integrated exponentials with proper handling of singular eigenvalues
        exp_w_1 = np.zeros_like(w)
        exp_w_1[w != 0] = (1-np.exp(-self.k*w[w!=0]*dt)) / (self.k*w[w!=0])
        exp_w_1[w == 0] = dt

        exp_w_2 = np.zeros_like(w)
        exp_w_2[w != 0] = (1-np.exp(-2*self.k*w[w!=0]*dt)) / (self.k*w[w!=0])
        exp_w_2[w == 0] = 2*dt

        # discrete noise correlation matrix Σ and its Cholesky decomposition
        if self.D > 0:
            Sig = V @ ((self.D * exp_w_2[:, None]) * V.T)
            LSig = scipy.linalg.cholesky(Sig, lower=True)
        else:
            Sig = np.zeros((self.N, self.N))
            LSig = np.zeros((self.N, self.N))

        self._dynamics = {
                'needs_updating' : False,
                'N'              : self.N,
                'D'              : self.D,
                'k'              : self.k,
                'dt'             : dt,
                'w'              : w,
                'mp_kw'          : mp_kw,
                'V'              : V,
                'B'              : B,
                'ss_CoD'         : ss_CoD,
                'exp_w_1'        : exp_w_1,
                'Sig'            : Sig,
                'LSig'           : LSig,
        }
        self.update_F_only()

    def update_F_only(self, override_full_update=False):
        """
        Update specifically the stuff depending on the external force F.

        Parameters
        ----------
        override_full_update : bool
            pretend that this was a full update. Useful when you changed only
            `!F` since the last update, to avoid recalculating everything else

        See also
        --------
        update_dynamics
        """
        if not np.any(self.F):
            G = np.zeros_like(self.F)
        else:
            V = self._dynamics['V']
            e = self._dynamics['exp_w_1']
            G = V @ (e[:, None] * (V.T @ self.F))
        
        self._dynamics['G'] = G
        self._dynamics['ss_M'] = self._dynamics['ss_CoD'] @ self.F

        if override_full_update:
            self._dynamics['needs_updating'] = False

    def check_dynamics(self, dt=None, run_if_necessary=True):
        """
        Check that dynamics are set up properly

        Parameters
        ----------
        dt : float or None
            the time step we should be set up for. Specify ``None`` to accept
            whatever was the value set up
        run_if_necessary : bool
            controls the behavior when dynamics are not set up. If set to
            ``True`` (default), `update_dynamics` is simply run with the
            settings we have. If set to ``False`` raise a ``RuntimeError`` if
            dynamics are not (correctly) set up for the current model.

        Raises
        ------
        RuntimeError
            if something is not set up properly

        See also
        --------
        update_dynamics
        """
        if dt is None:
            try:
                dt = self._dynamics['dt']
            except KeyError:
                raise RuntimeError("Call update_dynamics before running")

        try:
            if dt != self._dynamics['dt']:
                self._dynamics['needs_updating'] = True

            for key in ['N', 'D', 'k']:
                if self._dynamics[key] != getattr(self, key):
                    self._dynamics['needs_updating'] = True
        except KeyError:
            self._dynamics['needs_updating'] = True
        
        if self._dynamics['needs_updating']:
            if run_if_necessary:
                self.update_dynamics(dt)
            else:
                raise RuntimeError("Model changed since last call to update_dynamics()")

####### Propagation of an ensemble

    def steady_state(self):
        """
        Calculate the steady state ensemble

        Since our model is linear and driven by Gaussian noise, the steady
        state ensemble will always be Gaussian, and as such given by a mean
        vector and covariance matrix over all monomers.

        We use the Moore-Penrose pseudo-inverse of the connectivity matrix
        ``A`` to calculate the steady state distribution. This means that any
        degrees of freedom (such as center of mass for a free chain) will be
        pinned to the origin instead of diverging to infinity.

        Returns
        -------
        M : (N, d) np.ndarray, dtype=float
            mean position of each monomer
        C : (N, N) np.ndarray, dtype=float
            covariance between monomer coordinates in each dimension. Note that
            we assume independence of spatial dimensions, such that the full
            covariance ``C_ijab := <x_ia*x_jb>_c`` (with monomer indices i, j
            and spatial indices a, b) is written as the tensor product ``C_ijab
            = C_ij*δ_ab`` with Kronecker's δ. The matrix returned by this
            function is ``C_ij``.
        """
        try:
            self.check_dynamics(run_if_necessary=True)
        except RuntimeError: # if the model is really not set up, it won't have a time step, so we have to do this by hand
            self.update_dynamics()
        return self._dynamics['ss_M'], self._dynamics['ss_CoD']*self.D

    def propagate_M(self, M, dt=None, check_dynamics=True):
        """
        Propagate the mean of a Gaussian ensemble

        Parameters
        ----------
        M : (N, d) np.ndarray, dtype=float
            mean conformation
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        M : (N, d) np.ndarray, dtype=float
            like input `!M`, but propagated by `!dt`.

        See also
        --------
        propagate, propagate_C, steady_state
        """
        if check_dynamics:
            self.check_dynamics(dt, run_if_necessary=True)
        B = self._dynamics['B']
        G = self._dynamics['G']
        return B @ M + G

    def propagate_C(self, C, dt=None, check_dynamics=True):
        """
        Propagate the covariance of a Gaussian ensemble

        Parameters
        ----------
        C : (N, N) np.ndarray, dtype=float
            current covariance
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        C : (N, N) np.ndarray, dtype=float
            like input `!C`, but propagated by `!dt`.

        See also
        --------
        propagate, propagate_M, steady_state
        """
        if check_dynamics:
            self.check_dynamics(dt, run_if_necessary=True)
        B = self._dynamics['B']
        Sig = self._dynamics['Sig']
        return B @ C @ B + Sig

    def propagate(self, M, C, dt=None, check_dynamics=True):
        """
        Propagate a Gaussian ensemble one time step

        This is simply a wrapper for `propagate_M` and `propagate_C`.

        Parameters
        ----------
        M : (N, d) np.ndarray, dtype=float
            current mean conformation
        C : (N, N) np.ndarray, dtype=float
            current covariance
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        M, C
            like input, but propagated by `!dt`.

        See also
        --------
        propagate_M, propagate_C, steady_state
        """
        return self.propagate_M(M, dt, check_dynamics), \
               self.propagate_C(C, dt, check_dynamics=False) # if needed, dynamics were already checked in M step

####### Evolution of a single conformation

    def conf_ss(self):
        """
        Draw a conformation from steady state

        Returns
        -------
        (N, d) np.ndarray, dtype=float
            the drawn conformation

        See also
        --------
        steady_state
        """
        M, _ = self.steady_state()
        # C as returned by steady_state() might be singular, so utilize
        # analytical √C directly
        V = self._dynamics['V']
        mp_Dkw = self.D * self._dynamics['mp_kw']
        return M + V @ (np.sqrt(mp_Dkw)[:, None] * np.random.normal(size=(self.N, self.d)))

    def evolve(self, conf, dt=None, check_dynamics=True):
        """
        Evolve a conformation forward one time step

        Parameters
        ----------
        conf : (N, d) np.ndarray, dtype=float
            the conformation to start from
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        conf
            like input, but evolved for `!dt`.

        See also
        --------
        conf_ss
        """
        if check_dynamics:
            self.check_dynamics(dt, run_if_necessary=True)
        B = self._dynamics['B']
        L = self._dynamics['LSig']
        G = self._dynamics['G']
        return B @ conf + L @ np.random.normal(size=conf.shape) + G

####### MSD and related stuff

    def MSD(self, dts, w=None):
        r"""
        Calculate MSD for given degrees of freedom

        This is simply an evaluation of analytical expressions. Which path
        exactly we take depends on the system, see Notes.

        Parameters
        ----------
        dts : (T,) array-like, dtype=float
            the time lags to evaluate the MSD at. Entries should be greater or
            equal to zero
        w : (N,) np.ndarray, dtype=float
            measurement vector. Use this to specify the linear combination of
            monomers whose MSD you are interested in. See Examples. If
            unspecified, the function returns the full covariance matrix at lag
            Δt.

        Returns
        -------
        (T,) np.ndarray or (T, N, N) np.array
            the MSD, either as scalar function evaluated at `!dts` or matrix
            valued (if no measurement vector `!w` was specified).

        Notes
        -----
        The analytical solution for the MSD at time lag :math:`\Delta t` is
        given by

        .. math:: MSD(\Delta t) \equiv \left\langle\left(x(t+\Delta t) - x(t)\right)^{\otimes 2}\right\rangle = V\left[ 2D E + (\Delta t)^2 SV^TF F^TVS \right] V^T

        with :math:`E = \int_0^{\Delta t}\mathrm{d}\tau\exp(-kA\tau)`, ``S_ij =
        1 if i == j and λ_i = 0 else 0``, and :math:`A = V\Lambda V.T` is the
        eigendecomposition of A. Note that the Δt^2 terms just contributes
        ballistic motion for force components acting on singular degrees of
        freedom (e.g. center of mass).

        Examples
        --------
        >>> model = Model(N=25)
        ... dts = np.array([0, 1, 2, 10]) # some sample lag times
        ...
        ... # Calculate full matrices. This is usually not as relevant
        ... msd = model.MSD(dts)
        ...
        ... # MSD of the middle monomer
        ... w = np.zeros(model.N)
        ... w[12] = 1
        ... msd = model.MSD(dts, w)
        ...
        ... # MSD of the end-to-end vector
        ... w = np.zeros(model.N)
        ... w[0]  =  1
        ... w[-1] = -1
        ... msd = model.MSD(dts, w)
        ...
        ... # The end-to-end dof equilibrates, so we can calculate the limit
        ... # value
        ... msd_inf = model.MSD([np.inf], w)
        """
        dts = np.asarray(dts)
        if len(dts.shape) == 0:
            dts = dts[None] # make 1D
        if np.any(dts < 0):
            raise ValueError("dt should be >= 0")

        # Get pre-computed eigendecomposition
        # (Can't use `w` as eigenvalues, so those are `evs` now)
        evs = self._dynamics['w']
        evs_z = evs == 0
        evs_nz = ~evs_z

        V = self._dynamics['V']
        Vout = V
        if w is not None:
            Vout = w @ V
            Vout[np.abs(Vout) < 1e-10] = 0

        # Check time steps
        dt_finite = np.isfinite(dts)
        if np.any(dts[~dt_finite] != np.inf):
            raise ValueError(f"Invalid dt value(s): {dts[np.logical_and(~dt_finite, dts != np.inf)]}")
        if not np.all(dt_finite):
            if w is None:
                if np.any(evs_z):
                    raise ValueError("Cannot evaluate MSD(Δt = inf): System does not equilibrate")
            else:
                if np.any(Vout[..., evs_z] != 0):
                    raise ValueError("Cannot evaluate MSD(Δt = inf): w@System@w does not equilibrate")

        # Set up first part of the analytical expression
        k_evs_nz = self.k*evs[evs_nz]
        dDk_evs_nz = 2*self.d*self.D/k_evs_nz

        # Second part, applies only to singular degrees of freedom
        VF = (V.T @ self.F)
        VF[evs_nz, :] = 0 # action of S
        VSVF = Vout @ VF  # now (N, d) or (d,)
        Phi = VSVF @ VSVF.T
        if np.any(Phi):
            def add_Phi(VEV, dt):
                return VEV + (dt*dt)*Phi
        else:
            def add_Phi(VEV, dt):
                return VEV

        # Assemble
        # (this is not noticeably slower than list comprehension)
        msd = []
        for dt, finite in zip(dts, dt_finite):
            E = np.empty(len(evs), dtype=float)
            if finite:
                E[evs_nz] = (1-np.exp(-dt*k_evs_nz)) * dDk_evs_nz
                E[evs_z ] = 2*self.d*self.D*dt
            else:
                # We checked before that in this case the evs == 0 dof are
                # irrelevant anyways
                E[evs_nz] = dDk_evs_nz
                E[evs_z ] = 0

            if len(Vout.shape) > 1:
                out = Vout @ (E[:, None] * Vout.T)
            else:
                out = Vout @ (E * Vout)
            msd.append(add_Phi(out, dt))

        return np.array(msd)

    def ACF(self, dts, w=None):
        """
        Calculate autocovariances

        Exists mostly for completeness. Calculates the autocovariance as ``γ =
        exp(-k*A*dt) @ C``, where ``C`` is the steady state covariance as
        returned by `steady_state()`. Note that for any equilibrating degrees
        of freedom the ACF can also be calculated from the MSD as

            ACF(Δt) = 0.5*( MSD(∞) - MSD(Δt) )

        so sticking to the MSD is preferred. Any non-equilibrating dofs are
        pinned to zero by the Moore-Penrose inverse.

        The one benefit of this function over `MSD` is that it is faster
        (roughly 1.5x).

        Parameters
        ----------
        dts : (T,) array-like, dtype=float
            the time lags to evaluate the ACF at. Entries should be greater or
            equal to zero
        w : (N,) np.ndarray, dtype=float
            measurement vector. Use this to specify the linear combination of
            monomers whose ACF you are interested in. If unspecified, the
            function returns the full covariance matrix at lag Δt.

        Returns
        -------
        (T,) np.ndarray or (T, N, N) np.array
            the ACF, either as scalar function evaluated at `!dts` or matrix
            valued (if no measurement vector `!w` was specified).

        See also
        --------
        MSD
        """
        dts = np.asarray(dts)
        if len(dts.shape) == 0:
            dts = dts[None] # make 1D
        if np.any(dts < 0):
            raise ValueError("dt should be >= 0")

        # Get pre-computed eigendecomposition
        # (Can't use `w` as eigenvalues, so those are `evs` now)
        evs = self._dynamics['w']
        Vout = self._dynamics['V']
        if w is not None:
            Vout = w @ Vout

        # calculating Bs for all dt
        def calc_B(dt):
            if np.isfinite(dt):
                B = np.exp(-self.k*dt*evs)
            elif dt == np.inf:
                B = np.zeros_like(evs)
            else:
                raise ValueError(f"Invalid value: dt = {dt}")

            return B

        # assemble from steady state
        mp_dDkw = (self.d * self.D) * self._dynamics['mp_kw']
        if len(Vout.shape) > 1:
            return np.array([Vout @ ((calc_B(dt)*mp_dDkw)[:, None] * Vout.T) for dt in dts])
        else:
            return np.array([Vout @ (calc_B(dt) * mp_dDkw * Vout) for dt in dts])

####### Length & Time scales

    def timescales(self):
        """
        Give time scales relevant to the model

        Returns
        -------
        dict
            ``t_microscopic`` is where Rouse scaling (i.e. the model) starts to
            apply; ``t_Rouse`` is the classic Rouse time (relaxation of the
            slowest mode); ``t_equilibration`` is the actual crossover from
            Rouse to equilibrium of the end-to-end distance (for a free chain)
        """
        return {
                't_microscopic' : 1/self.k,
                't_Rouse' : (self.N / np.pi)**2 / self.k,                
                't_equilibration' : np.pi*self.N**2 / (4*self.k),
                }

    def Gamma(self):
        """
        MSD prefactor of a single locus on the polymer
        
        Returns
        -------
        float
        """
        return 2*self.d*self.D / np.sqrt(np.pi*self.k)

    def rms_Ree(self, L=None):
        """
        RMS end-to-end distance for chain of length L

        Parameters
        ----------
        L : float, optional
            number of (effective) bonds in the chain. Defaults to ``self.N-1``.

        Returns
        -------
        float
        """
        if L is None:
            L = self.N-1
        return np.sqrt(self.d*self.D/self.k * L)

####### Auxiliary things

    def contact_frequency(self):
        """
        Calculate a contact frequency matrix for the system

        This is intended to produce HiC-like maps for the system. It is based
        on the mean squared distance between each pair of loci, and then
        converts to contact frequency by a scaling exponent of ``-self.d/2``.
        Note that these are not actual probabilities, but relative frequencies

        Returns
        -------
        (N, N) np.ndarray, dtype=float
        """
        _, J = self.steady_state()
        Jii = np.tile(np.diagonal(J), (len(J), 1))
        with np.errstate(divide='ignore'): # the diagonal gives np.inf; that's fine
            return (Jii + Jii.T - 2*J)**(-self.d/2)

def twoLocusMSD(dts, Gamma, J):
    r"""
    MSD for relative position of two loci on infinite continuous Rouse polymer

    The analytical expression is given by

    .. math:: MSD(\Delta t) = 2\Gamma\sqrt{\Delta t}\left(1 - \exp\left(-\frac{\tau}{\Delta t}\right)\right) + 2J\mathrm{erfc}\sqrt{\frac{\tau}{\Delta t}}\,,

    with :math:`\tau\equiv \frac{1}{\pi}\left(\frac{J}{\Gamma}\right)^2`.

    Parameters
    ----------
    dts : array-like, dtype=float
        the times at which to evaluate the MSD
    Gamma : float
        the prefactor for the polymer part (0.5 scaling) of the MSD. Note that
        this parameter should be the prefactor for tracking of one locus, i.e.
        the MSD produced by this function (which is the distance between two
        loci) will have a prefactor of ``2*Gamma``.
    J : float
        the mean squared distance of the two loci at steady state. This is half
        the asymptotic value of the MSD: ``MSD(Δt --> inf) = 2*J``.

    Returns
    -------
    (T,) np.array
        the MSD evaluated at times `!dts`
    """
    dts = np.asarray(dts)
    if len(dts.shape) == 0:
        dts = dts[None]
    ind_zero = dts == 0
    ind_finite = np.logical_and(np.isfinite(dts), dts > 0)
    ind_inf = dts == np.inf

    ind_invalid = (~ind_zero * ~ind_finite * ~ind_inf).astype(bool)
    if np.any(ind_invalid):
        raise ValueError(f"dts contain invalid values: {dts[ind_invalid]}")

    tau = (J / Gamma)**2 / np.pi
    ret = np.empty(dts.shape, dtype=float)
    ret[ind_zero]   = 0
    with np.errstate(under='ignore'):
        ret[ind_finite] = 2*Gamma * np.sqrt(dts[ind_finite]) * (1 - np.exp(-tau/dts[ind_finite])) \
                            + 2*J * scipy.special.erfc( np.sqrt(tau/dts[ind_finite]) )
    ret[ind_inf]    = 2*J
    return ret

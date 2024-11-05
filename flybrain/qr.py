import torch


class QR(torch.autograd.Function):
    """
    Differentiable QR decomp implementation based on the work of
    References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018).
    Auto-Differentiating Linear Algebra.
    """

    @staticmethod
    def forward(self, A):
        Q, R = torch.linalg.qr(A)
        self.save_for_backward(A, Q, R)
        return Q, R

    @staticmethod
    def backward(self, dq, dr):
        A, q, r = self.saved_tensors
        if r.shape[0] == r.shape[1]:

            return _simple_qr_backward(q, r, dq, dr)
        M, N = r.shape
        B = A[:, M:]
        dU = dr[:, :M]
        dD = dr[:, M:]
        U = r[:, :M]

        da = _simple_qr_backward(q, U, dq + B @ dD.t(), dU)
        db = q @ dD
        return torch.cat([da, db], 1)


def _simple_qr_backward(q, r, dq, dr):
    if r.shape[-2] != r.shape[-1]:
        raise NotImplementedError(
            "QrGrad not implemented when ncols > nrows "
            "or full_matrices is true and ncols != nrows."
        )

    qdq = q.t() @ dq
    qdq_ = qdq - qdq.t()
    rdr = r @ dr.t()
    rdr_ = rdr - rdr.t()
    tril = torch.tril(qdq_ + rdr_)

    def _TriangularSolve(x, r):
        """Equiv to x @ torch.inverse(r).t() if r is upper-tri."""
        res = torch.linalg.solve_triangular(r, x.t(), upper=True).t()
        return res

    grad_a = q @ (dr + _TriangularSolve(tril, r))
    grad_b = _TriangularSolve(dq - q @ qdq, r)

    return grad_a + grad_b

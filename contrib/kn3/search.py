
def run_search(burst,access):
    """

    Execute the burst nearest neighbors search

    """

    # -- create resources --
    device = "cuda:0"
    res = faiss.StandardGpuResources()


    # -- create dummy data --
    tf32,ti32 = th.float32,th.int32
    xb = th.zeros((10,10),device=device,dtype=tf32)
    xq = th.zeros((10,3),device=device,dtype=ti32)
    k = 3
    D,I = None,None

    # -- check sizes --
    nb, d = xb.size()
    nq, d2 = xq.size()

    # -- alloc/format bufs --
    D = get_buf(D,nq,k,device,tf32)
    I = get_buf(I,nq,k,device,ti32)

    # --- faiss info --
    xb_row_major = check_contiguous(xb)
    xq_row_major = check_contiguous(xq)
    assert xq_row_major is True,"Must be row major."
    xb_ptr,xb_type = get_float_ptr(xb)
    xq_ptr,xq_type = get_int_ptr(xq)
    I_ptr,I_type = get_int_ptr(I)
    D_ptr,D_type = get_float_ptr(D)

    # -- create args --
    args = faiss.GpuKn3DistanceParams()
    args.fxn_name = faiss.Kn3FxnName_FILL
    args.k = k
    args.dims = d
    args.srch_burst = xb_ptr
    args.vectorType = xb_type
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.fill_dists = val_dists
    args.fill_inds = val_inds

    # -- exec --
    with using_stream(res):
        faiss.bfKn3(res, args)

    return D,I

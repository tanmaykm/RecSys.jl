using RecSys
include("/home/tan/Work/julia/packages/RecSys/src/chunk.jl")

function split_sparse(S, chunkmax, filepfx)
    metafilename = "$(filepfx).meta"
    open(metafilename, "w") do mfile
        chunknum = 1
        count = 1
        colstart = 1
        splits = UnitRange[]
        nzval = S.nzval
        rowval = S.rowval
        println("max cols: $(size(S,2))")
        for col in 1:size(S,2)
            npos = S.colptr[col+1]
            if (npos >= (count + chunkmax)) || (col == size(S,2))
                print("\tchunk $chunknum ... ")
                cfilename = "$(filepfx).$(chunknum)"
                println(mfile, colstart, ",", col, ",", cfilename)
                open(cfilename, "w") do cfile
                    for cidx in colstart:col
                        rowstart = S.colptr[cidx]
                        rowend = S.colptr[cidx+1]-1
                        for ridx in rowstart:rowend
                            println(cfile, rowval[ridx], ",", cidx, ",", nzval[ridx])
                        end
                    end
                end
                #    SC = sub(S, :, colstart:(col-1))
                #    R,C,NZ = findnz(SC)
                #    print("... ")
                #    for idx in 1:length(R)
                #        println(cfile, R[idx], ",", C[idx], ",", NZ[idx])
                #    end
                #end
                push!(splits, colstart:col)
                colstart = col+1
                count = npos
                chunknum += 1
                println("done")
            end
        end
        println("splits: $splits")
    end
    nothing
end

function splitall(inp::DlmFile, output_path::AbstractString, nsplits::Int)
    println("reading inputs...")
    ratings = RecSys.read_input(inp)

    users   = convert(Vector{Int64},   ratings[:,1]);
    items   = convert(Vector{Int64},   ratings[:,2]);
    ratings = convert(Vector{Float64}, ratings[:,3]);
    R = sparse(users, items, ratings);
    R, item_idmap, user_idmap = RecSys.filter_empty(R)
    isempty(item_idmap) || println("item ids were re-mapped")
    isempty(user_idmap) || println("user ids were re-mapped")

    nratings = length(ratings)
    nsplits_u = round(Int, nratings/nsplits)
    nsplits_i = round(Int, nratings/nsplits)

    println("splitting R itemwise at $nsplits_i items...")
    split_sparse(R, nsplits_i, joinpath(output_path, "R_itemwise"))
    RT = R'
    println("splitting RT userwise at $nsplits_u users...")
    split_sparse(RT, nsplits_u, joinpath(output_path, "RT_userwise"))
    nothing
end

function split_movielens(dataset_path = "/data/Work/datasets/movielens/ml-20m")
    ratings_file = DlmFile(joinpath(dataset_path, "ratings.csv"); dlm=',', header=true)
    splitall(ratings_file, joinpath(dataset_path, "splits"), 10)
end

function split_lastfm(dataset_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005")
    ratings_file = DlmFile(joinpath(dataset_path, "user_artist_data.txt"))
    splitall(ratings_file, joinpath(dataset_path, "splits"), 20)
end

function load_splits(dataset_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits")
    cf = ChunkedSparseMatrixCSC(joinpath(dataset_path, "R_itemwise.meta"))
    println(cf)
    nchunks = length(cf.chunks)
    for idx in 1:10
        cid = floor(Int, nchunks*rand()) + 1
        println("fetching from chunk $cid")
        c = cf.chunks[cid]
        key = floor(Int, length(c.keyrange)*rand()) + c.keyrange.start
        println("fetching key $key")
        r,v = data(cf, key)
        #println("\tr:$r, v:$v")
    end
    println("finished")
end

function create_memmapped_splits(dataset_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splitsmem")
    metafilename = joinpath(dataset_path, "mem.meta")
    N = 3
    M = 100
    sz = M*N*sizeof(Float64)
    NC = 10

    touch(metafilename)
    cf = ChunkedMemMappedMatrix(metafilename, N)

    for idx in 1:NC
        chunkfname = joinpath(dataset_path, "mem.$idx")
        r1 = (idx-1)*M+1
        r2 = idx*M
        chunk = Chunk(chunkfname, 0, sz, r1:r2, MemMappedMatrix{Float64,N})
        push!(cf.chunks, chunk)
    end
    for idx in 1:NC
        A = data(cf, idx*M)
        fill!(A, idx)
    end
    for idx in 1:NC
        A = data(cf, idx*M)
        println(A[1])
    end

    writemeta(cf)
    cf = ChunkedMemMappedMatrix(metafilename, N)
    for idx in 1:NC
        A = data(cf, idx*M)
        println(A[1])
    end
end

split_movielens()
#load_splits()
#create_memmapped_splits()

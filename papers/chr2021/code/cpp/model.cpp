#include "types.h"
#include "helpers.h"

void mulAlphaCitmask(matrixf &alpha, const matrixf &citmask)
{
	const size_t rows = alpha.size(), cols = ncols<float>(alpha);
	if(rows!=citmask.size() || cols!=ncols<float>(citmask)){
		COUT << "wrong dimensions in mulAlphaCitmask" << std::endl;
		return;
	}
	for(size_t r=0;r<rows;++r){
		for(size_t c=0;c<cols;++c){
			if(r!=c){
				const float v = (citmask[r][c]==0) ? 1e-4f : citmask[r][c];
				alpha[r][c]*=v;
			}
		}
	}
}

void cgsMmCitationNgram(const std::vector<inttype> &Doc, const std::vector<inttype> &Fea, 
	std::vector<inttype> &Cit, std::vector<inttype> &Tim, std::vector<inttype> &Big,
	const matrixf &tau, const matrixf &citmask,
	const std::string &respath, const size_t nItrs,
	const std::map<std::string,float> &params,
	const inttype testId,
	const bool recordTrainAndTest)
{
	COUT << "Hi from ToCN, running for " << nItrs << " epochs" << std::endl;
	if(testId > 0){
		COUT << "test with id=" << testId << std::endl;
	}
	std::map<std::string,std::vector<inttype>> emptyRes;

	std::random_device rd;
	std::mt19937 gen(rd());

	const inttype T = 1 + *std::max_element(Tim.begin(),Tim.end()),
		D = 1 + *std::max_element(Doc.begin(),Doc.end()),
		V = 1 + *std::max_element(Fea.begin(),Fea.end());

	const bool useCitMask = (bool)getParam(params, "useCitMask", 1);
	const bool sampleAtLastItr = (bool)getParam(params, "sampleAtLastItr", 0);
	const float scaleAlpha = 1.f;
	matrixf alpha = reestimateAlpha(Cit,Tim,D,T, scaleAlpha);
	if(useCitMask){
		mulAlphaCitmask(alpha,citmask);
	}
	////// priors
	// values set using PPCs
	const float gamma = getParam(params,"gamma", 0.5f),
		delta = getParam(params,"delta", 0.01f);
	COUT << "gamma: " << gamma << ", delta: " << delta << ", citmask: " << useCitMask << ", sampleAtLastItr: " << sampleAtLastItr << std::endl;
	matrixi A = matrix<inttype>(D,D), // text -> cited
		D1 = matrix<inttype>(V,V),    // words i and i+1 form a bigram
		D0 = matrix<inttype>(V,V),    // ... do not form a bigram
		EvidenceU = matrix<inttype>(V,D); // text -> feature, transposed
	matrixf Cu_t = matrix<float>(V,T);    // time -> unigram feature for word 1, transposed
	std::vector<float*> B = aligned_matrix<float>(D,T);     // text -> time
	
	std::vector<inttype> multiBigramIxes;
	std::map<std::pair<inttype,inttype>,inttype> big2ix;
	if(!buildEvidenceBigrams(Doc,Fea,multiBigramIxes,big2ix)){
		return;
	}
	std::vector<int> Cb ((big2ix.size()+1)*T, 0);
	std::vector<unsigned char> EvidenceBi( (big2ix.size()+1)*D, 0x00);
	matrixf Cbs_t = matrix<float>(V,T);
	for(size_t i=0;i<Doc.size()-1;++i){
		const inttype d = Doc[i], 
			v1 = Fea[i], v2 = Fea[i+1],
			c = Cit[i], t = Tim[i],
			biIx = multiBigramIxes[i];
		inttype b = Big[i];
		if(biIx==0){
			// this is not a valid multi-text bigram
			b = 0;
			Big[i] = 0;
		}else{
			EvidenceBi[INDEX2(d,biIx,D)] = 0x01;
		}
		
		++A[d][c];
		++B[c][t];
		if(b==1){ // bigram
			Cb[INDEX2(t,biIx,T)]+=1;
			++Cbs_t[v1][t];
			++D1[v1][v2];
		}else{
			++Cu_t[v1][t];
			++D0[v1][v2];
		}
		// static evidence for filtering appropriate candidates
		++EvidenceU[v1][d];
	}
	
	float* Bs = aligned_rowsums<float>(B, T);
	std::vector<float> taus = rowsums<float>(tau);
	float* Cus = vec2ptr<float>(colsums<float>(Cu_t));
	
	std::vector<inttype> ixes(Doc.size()-1); // skip the last word
	std::iota(std::begin(ixes), std::end(ixes), 0);
	std::uniform_real_distribution<float> dis(0.0, 1.0);
	const size_t allocSz = D*T*2;
	inttype * ccc = new inttype[allocSz], 
		*ttt = new inttype[allocSz],
		*bbb = new inttype[allocSz];

	std::vector<float> p(allocSz,0.f), pcs;
	// for each text (row): first and last valid time slot according to tau
	std::vector<std::pair<int,int>> txt2slot = getAllowedTimeSlots(tau);
	if(txt2slot.empty()){
		return;
	}
	// needed for ppc!
	write_data(std::vector<std::vector<inttype>>({ Fea, multiBigramIxes }), 
		std::vector<std::string>({ "Fea", "BigIx"  }), 
		"../../../../data/latin/output/citation-mm/ppc-num-slots/multi-bigrams.dat", 1);
	const size_t coutItr = 10;
	std::vector<inttype> Doc_all, Fea_all, Cit_all, Tim_all, Big_all, Itr_all;
	for(size_t trainTestItr = 1; trainTestItr<=2;++trainTestItr)
	{
		const size_t itrs = (trainTestItr==1) ? nItrs : 50;
		for(size_t itr=0;itr<itrs;++itr)
		{
			// re-estimate the citation structure
			alpha = reestimateAlpha(Cit,Tim,D,T, scaleAlpha);
			if(useCitMask){
				mulAlphaCitmask(alpha,citmask);
			}
			std::random_shuffle(ixes.begin(), ixes.end());
			int chTim = 0, chCit = 0, chBig = 0;
			std::chrono::steady_clock::time_point tstart = std::chrono::steady_clock::now();
			for(size_t ix : ixes)
			{
				const inttype d = Doc[ix], v1 = Fea[ix], v2 = Fea[ix+1], c = Cit[ix], t = Tim[ix], b = Big[ix],
					biIx = multiBigramIxes[ix];
				if(trainTestItr==1 && d==testId-1){
					continue;
				}
				if(Doc[ix+1]!=d){
					// both words must be from the same text
					continue;
				}
				--A[d][c];
				--B[c][t];
				--Bs[c];
				if(b==1){
					--Cb[INDEX2(t,biIx,T)];
					--Cbs_t[v1][t];
					--D1[v1][v2];
				}else{
					--Cu_t[v1][t];
					--Cus[t];
					--D0[v1][v2];
				}
				// new values
				size_t n=0; // counter for the results
				const bool checkBigrams = (biIx>0); 
				
				const float pd1 = (float(D1[v1][v2]) + delta), 
					pd0 = (float(D0[v1][v2])+delta); 
				
				std::allocator_traits<std::vector<inttype>>::const_pointer pE1 = &EvidenceU[v1][0], pE2 = &EvidenceU[v2][0];
				std::allocator_traits<std::vector<float>>::const_pointer pConn = &alpha[d][0];
				for(inttype cc=0;cc<D;++cc) // for each possible source of citations
				{
					// the source must be earlier and contain the first word
					if(pConn[cc]==0 || pE1[cc]==0  || (trainTestItr==1 && cc==testId-1)){
						continue;
					}
					const bool allowBigram = (checkBigrams && EvidenceBi[INDEX2(cc,biIx,D)] );
					const float pa = float(A[d][cc]) + alpha[d][cc];
					const float divB = Bs[cc] + taus[cc];
					const float * pB = B[cc];
					std::allocator_traits<std::vector<float>>::const_pointer pCu = &Cu_t[v1][0];
					std::allocator_traits<std::vector<float>>::const_pointer tauc = &tau[cc][0],
						pCbs = &Cbs_t[v1][0];
					const int start = txt2slot[cc].first, end = txt2slot[cc].second; // possible temporal range
					for(inttype tt=start;tt<end;++tt)
					{
						const float pb = ( pB[tt] + tauc[tt])/divB;
						// unigram ...
						const float pc0 = ( pCu[tt] + gamma)/( Cus[tt] + float(V)*gamma);
						p[n] = pa * pb * pc0 * pd0;
						ccc[n] = cc;
						ttt[n] = tt;
						bbb[n] = 0;
						++n;
						// or bigram?
						if(allowBigram){
							const float pc1 = ( Cb[ INDEX2(tt,biIx,T) ] + gamma) / ( pCbs[tt] + float(V)*gamma);
							p[n] = pa * pb * pc1 * pd1;
							ccc[n] = cc;
							ttt[n] = tt;
							bbb[n] = 1;
							++n;
						}
					}
				}

				// sampling
				pcs.resize(n);
				std::partial_sum(p.begin(), p.begin() + (size_t)n, pcs.begin(), std::plus<float>());// cumsum
				float r = dis(gen), highest = pcs.back();
				std::allocator_traits<std::vector<float>>::const_pointer cpcs = &pcs[0];
				bool sampled = false;
				for(size_t w = 0; w<n; ++w)
				{
					float v = (*cpcs++) / highest;
					if(v >= r)
					{
						const inttype tnew = ttt[w], cnew = ccc[w], bnew = bbb[w];

						A[d][cnew]++;
						B[cnew][tnew]++;
						Bs[cnew]++;
						if(bnew==1){
							++Cb[INDEX2(tnew,biIx,T)];
							++Cbs_t[v1][tnew];
							++D1[v1][v2];
						}else{
							++Cu_t[v1][tnew];
							++Cus[tnew];
							++D0[v1][v2];
						}
						Tim[ix] = tnew;
						Cit[ix] = cnew;
						Big[ix] = bnew;
						if(c!=cnew){++chCit;}
						if(t!=tnew){++chTim;}
						if(b!=bnew){++chBig;}
						sampled = true;
						break;
					}
				}
				if(!sampled){
					COUT << "not sampled" << std::endl;
					print<float>(std::vector<float>(p.begin(), p.begin() + (size_t)n));
					return;
				}
			}
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			/*sanity and clean-up*/
			sanityCheckCountMatrix<inttype>(A,"A");
			sanityCheckCountMatrix(B, T,"B");
			sanityCheckCountMatrix<inttype>(D1,"D1");
			sanityCheckCountMatrix<float>(Cu_t,"Cu_t");
			sanityCheckCountMatrix<inttype>(D0,"D0");
			sanityCheckCountMatrix<float>(Cbs_t,"Cbs_t");
			int cbwarn = 0;
			for(size_t u=0;u<Cb.size();++u){
				if(Cb[u]<0){
					++cbwarn;
				}
			}
			if(cbwarn>0){
				COUT << cbwarn << " values in Cb <0" << std::endl;
			}
			size_t total1 = 0, total2 = 0, occupied = 0;
			if(itr % coutItr==0){
				COUT << "-- iteration " << (itr+1) << ", " << 
					total1 << "->" << total2 << " items, " << std::chrono::duration_cast<std::chrono::milliseconds>(end - tstart).count() << " ms, " << 
					"chCit: " << chCit << ", chTim: " << chTim << ", chBig: " << chBig << std::endl;
			}
			if(sampleAtLastItr==true)
			{
				if(itr==itrs-1){
					COUT << "sampling from train at the last iteration ..." << std::endl;
					std::vector<inttype> itrs(Doc.size(), itr);
					write_data(std::vector<std::vector<inttype>>({ Doc,Fea,Cit,Tim,Big, itrs }), 
						std::vector<std::string>({ "Doc", "Fea", "Cit", "Tim", "Big", "Itr"  }), 
						respath, 1);
				}
			}
			else{
				if(testId==0)
				{
					if(shouldSample2(itr,itrs)){
						COUT << "sampling from train (final) ..." << std::endl;
						Doc_all.insert(Doc_all.end(), Doc.begin(), Doc.end());
						Fea_all.insert(Fea_all.end(), Fea.begin(), Fea.end());
						Cit_all.insert(Cit_all.end(), Cit.begin(), Cit.end());
						Tim_all.insert(Tim_all.end(), Tim.begin(), Tim.end());
						Big_all.insert(Big_all.end(), Big.begin(), Big.end());
						std::vector<inttype> itrs(Doc.size(), itr);
						Itr_all.insert(Itr_all.end(), itrs.begin(), itrs.end());
						write_data(std::vector<std::vector<inttype>>({ Doc_all,Fea_all,Cit_all,Tim_all,Big_all, Itr_all }), 
							std::vector<std::string>({ "Doc", "Fea", "Cit", "Tim","Big","Itr"  }), 
							respath, 1);
					}else if((itr+1) % 25==0){// intermediate results
						 COUT << "sampling from train (intermediate) ..." << std::endl;
						 std::vector<inttype> itrs(Doc.size(), itr);
						 write_data(std::vector<std::vector<inttype>>({ Doc,Fea,Cit,Tim,Big, itrs }), 
						 std::vector<std::string>({ "Doc", "Fea", "Cit", "Tim", "Big", "Itr"  }), 
							respath, 1);
					}
				}
				else if(trainTestItr==2 && shouldSample2(itr,itrs))
				{
					if(recordTrainAndTest==true){
						COUT << "sampling from train AND test ..." << std::endl;
						Doc_all.insert(Doc_all.end(), Doc.begin(), Doc.end());
						Fea_all.insert(Fea_all.end(), Fea.begin(), Fea.end());
						Cit_all.insert(Cit_all.end(), Cit.begin(), Cit.end());
						Tim_all.insert(Tim_all.end(), Tim.begin(), Tim.end());
						Big_all.insert(Big_all.end(), Big.begin(), Big.end());
						std::vector<inttype> itrs(Doc.size(), itr);
						Itr_all.insert(Itr_all.end(), itrs.begin(), itrs.end());
					}else{
						// recording values only from the test set
						COUT << "sampling from test ..." << std::endl;
						for(size_t i=0;i<Doc.size();++i)
						{
							if(Doc[i]==(testId-1)){
								Doc_all.push_back(Doc[i]);
								Fea_all.push_back(Fea[i]);
								Cit_all.push_back(Cit[i]);
								Tim_all.push_back(Tim[i]);
								Big_all.push_back(Big[i]);
								Itr_all.push_back(itr);
							}
						}
					}
					write_data(std::vector<std::vector<inttype>>({ Doc_all,Fea_all,Cit_all,Tim_all,Big_all, Itr_all }), 
						std::vector<std::string>({ "Doc", "Fea", "Cit", "Tim","Big", "Itr"  }), 
						respath, 1);
					writeMatrix<float>(alpha, respath + ".alpha");
				}
			}
		}// trainTestItr
		if(testId==0){
			// no testing, only one main loop
			break;
		}
	}
	delete [] Cus;
	delete [] ccc;
	delete [] ttt;
	delete [] bbb;
	delete_aligned_matrix<float>(B);
	_aligned_free(Bs);
}

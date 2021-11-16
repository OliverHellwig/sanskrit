#include "types.h"
#include "helpers.h"

matrixf onehot(const std::vector<inttype> &init)
{
	size_t N = init.size();
	size_t C = std::set<inttype>(init.begin(), init.end()).size(); // column number
	matrixf m = matrix<float>(N, C);
	for (size_t n = 0; n<N; n++) {
		m[n][init[n]] = 1.f;
	}
	return m;
}

std::pair<matrixf, std::vector<inttype>> onehot_random(const inttype rows, const inttype cols)
{
	matrixf m = matrix<float>(rows, cols);
	std::vector<inttype> v(rows);
	std::default_random_engine gen;
	std::uniform_int_distribution<int> dis(0, int(cols)-1);
	for(inttype row=0;row<rows;row++){
		int num = dis(gen);
		m[row][num] = 1.f;
		v[row] = num;
	}
	return std::make_pair(m,v);
}

std::map<inttype,int> table(const std::vector<inttype> &v){
	std::map<inttype,int> m;
	for(auto i : v){
		auto it = m.find(i);
		if(it==m.end()){
			m[i] = 1;
		}else{
			it->second++;
		}
	}
	return m;
}

/*
	\param init Values in the colums
	\param groups Values in the rows
*/
matrixi table(const std::vector<inttype> &init, const std::vector<inttype> &groups)
{
	size_t R = 1 + *std::max_element(groups.begin(), groups.end()),
		C = 1 + *std::max_element(init.begin(), init.end()); // column number
	matrixi m = matrix<inttype>(R,C);
	for(size_t n=0;n<init.size();++n){
		++m[groups[n]][init[n]];
	}
	return m;
}



std::vector<inttype> sample(const size_t N, const int maxval)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<int> dis(0,maxval);
	assert(false); // read the docs! the initialization is wrong!
	std::vector<inttype> t(N);
	for(size_t n=0;n<N;n++){
		t[n] = dis(gen);
	}
	return t;
}

void write_data(const std::vector<std::vector<inttype>> &d, const std::vector<std::string> &headers, const std::string &respath, const inttype increment/*=1*/)
{
	std::cout << "Writing to file " << respath << std::endl;
	std::ofstream file(respath, std::ios::binary);
	if(!file.good()){
		std::cout << " FAILURE!" << std::endl;
		return;
	}
	for(msztype j=0;j<headers.size();j++){
		if(j>0){
			file << " " ;
		}
		file << headers[j];
	}
	file << std::endl;
	msztype N = d[0].size(), // columns
		M = d.size(); // rows
					  // see the notes on param d for the arrangement of the iterations!
	for(msztype n=0;n<N;n++){
		for(msztype m=0;m<M;m++){
			if(m>0){ file << " ";}
			file << d[m][n]+increment;
		}
		file << std::endl;
	}
	file.close();
}



std::string getPathAffix()
{
	std::string pathAffix;
	COUT << "Enter the path affix = data type. Possible values" << std::endl <<
		" * withStop-withFrequent-exactDates/dateRanges" << std::endl <<
		" * noStop-noFrequent-withCitations/noCitations-exactDates/dateRanges" << std::endl <<
		" ... and any combination of them" << std::endl;
	std::cin >> pathAffix;
	if(pathAffix.empty()){
		std::cout << "No path affix given" << std::endl;
		return "";
	}
	return "-" + pathAffix ;
}

std::pair<float,float> meanAndVarianceRaw(const std::vector<float> &v)
{
	float m = 0.f, var = 0.f;
	size_t n=v.size();
	if(n > 0)
	{
		std::allocator_traits<std::vector<float>>::const_pointer pv = &v[0];
		if(n>0){
			for(size_t i=0;i<n;++i){
				m+=pv[i];
			}
			m/=float(n);
	
			for(size_t i=0;i<n;++i){
				var+=(pv[i]-m) * (pv[i]-m);
			}
			var/=float(n);
		}
	}
	return std::make_pair(m,var);
}

/*
imposes the hard constraints resulting from the temporal priors
*/
std::vector<std::pair<int,int>> getAllowedTimeSlots(const matrixf &tau)
{
	std::vector<std::pair<int,int>> txt2slot;
	for(size_t i=0;i<tau.size();++i){
		int start=-1, end=-1;
		for(size_t j=0;j<tau.at(i).size();++j){
			if(tau[i][j]>0 && start==-1){
				start = (int)j;
			}else if(tau[i][j]==0 && start>-1 && end==-1){
				end = (int)(j);
			}
		}
		if(end==-1 && start>-1){
			end = int(tau.at(i).size());
		}
		if(start>-1 && end>-1){
			txt2slot.push_back(std::make_pair(start,end));
		}else{
			COUT << "wrong start/end at index " << i << std::endl;
			return std::vector<std::pair<int,int>>();
		}
	}
	return txt2slot;
}

std::vector<std::vector<inttype>> getCitableTexts(const matrixi &conn)
{
	std::vector<std::vector<inttype>> citable;
	for(size_t i=0;i<conn.size();++i){
		std::vector<inttype> v;
		for(size_t j=0;j<conn.at(i).size();++j){
			if(conn[i][j]>0){
				v.push_back((inttype)j);
			}
		}
		if(v.empty()){
			COUT << "no citable texts for " << i << std::endl;
			return std::vector<std::vector<inttype>>();
		}
		citable.push_back(v);
	}
	return citable;
}

matrixf conn2alpha(const matrixi &conn)
{
	matrixf alpha = matrix<float>(conn.size(),conn.begin()->size());
	float maxconn = (float)maxElement<inttype>(conn);
	for(size_t i=0;i<conn.size();++i){
		for(size_t j=0;j<conn[0].size();++j){
			alpha[i][j] = float(conn[i][j])/maxconn;
		}
	}
	return alpha;
}

/*
prior for models that adapt their citation structure to the predicted times
\return The prior matrix for document -> citation
*/
matrixf reestimateAlpha(const std::vector<inttype> &cits, const std::vector<inttype> &tim, const inttype D, const inttype T, const float scaleAlpha)
{
	std::vector<std::vector<float>> v((size_t)D);
	for(size_t i=0;i<cits.size();++i){
		v[cits[i]].push_back(float(tim[i]));
	}
	auto mv = std::vector<std::pair<float,float>>(D);
	for(size_t d=0;d<v.size();++d){
		mv[d] = meanAndVarianceRaw(v.at(d));
	}
	/*limit for (non-)citing; this corresponds to approx. 90 yrs in the initial 55 setting*/
	float lim = 3.f/55.f * float(T);
	matrixf alpha = matrix<float>(D,D);
	const float autoCit = 10.f;
	for(inttype d=0;d<D;++d){
		float sum = 0.f;
		float md = std::max(mv[d].first, 1e-6f);
		for(inttype c=0;c<D;++c){
			if(d==c){
				alpha[d][c] = autoCit;
			}else{
				const float mc = mv[c].first;
				const float est = (mc - md > lim) ? 0.f : 1.f/(1+std::exp(-(md-mc)));
				alpha[d][c] = est;
			}
			alpha[d][c]*=scaleAlpha;
			sum+=alpha[d][c];
		}
		if(sum==0 || std::isnan(sum) || std::isinf(sum)){
			COUT << "alpha failure with D=" << D << std::endl;
			print<float>(alpha[d]);
			for(size_t k=0;k<mv.size();++k){
				COUT << (k+1) << ": " << mv[k].first << " " << mv[k].second << std::endl;
			}
		}
	}
	return alpha;
}

std::vector<float> readGoldTimeSlots()
{
	std::ifstream f("../data/input/gold-time-slots.dat", std::ios::binary);
	std::string line;
	std::vector<float> goldTimeSlots;
	while(std::getline(f,line)){
		while (!line.empty() && (line.back() == 10 || line.back() == 13)) {
			line = line.substr(0, line.size() - 1);
		}
		if(!line.empty()){
			goldTimeSlots.push_back(std::stoi(line));
		}
	}
	f.close();
	return goldTimeSlots;
}


std::vector<maptype> buildEvidenceBigrams(const std::vector<inttype> &Doc, const std::vector<inttype> &Fea)
{
	const inttype D = 1 + *std::max_element(Doc.begin(),Doc.end()),
		V = 1 + *std::max_element(Fea.begin(),Fea.end());
	std::vector<maptype> E(D*V);
	std::allocator_traits<std::vector<inttype>>::const_pointer docs = &Doc[0], f = &Fea[0];
	for(size_t i=0;i<Doc.size()-1;++i){
		const inttype d = *docs++, w1 = f[i], w2 = f[i+1];
		incbi(E[INDEX2(d,w1,D)],w2,1);
	}
	return E;
}

/*
this is the good function
*/
bool buildEvidenceBigrams(const std::vector<inttype> &Doc, const std::vector<inttype> &Fea,
	std::vector<inttype> &multiBigramIxes, std::map<std::pair<inttype,inttype>,inttype> &big2ix)
{
	
	const inttype D = 1 + *std::max_element(Doc.begin(),Doc.end()),
		V = 1 + *std::max_element(Fea.begin(),Fea.end());
	matrixi bitmp = std::vector<std::vector<inttype>>(V);
	for(inttype w=0;w<V;++w){
		bitmp.at(w) = std::vector<inttype>(V,-1);
	}
	// does the bigram v1,v2 occur in more than one text? If yes, multiBigrams[v1][v2]==1.
	std::vector<std::vector<unsigned char>> multiBigrams = matrix<unsigned char>(V,V);
	int prevDoc = -1;
	for(size_t i=0;i<Doc.size()-1;++i){
		const inttype d = Doc[i], 
			v1 = Fea[i], v2 = Fea[i+1];
		if(!(d>=prevDoc)){
			COUT << "the documents must be in increasing order!" << std::endl;
			return false;
		}
		prevDoc = d;
		const inttype curBigramDoc = bitmp[v1][v2];
		if(curBigramDoc!=d){ // new document for this bigram
			if(curBigramDoc>d){
				COUT << "the documents must be in increasing order!" << std::endl;
				return false;
			}
			bitmp[v1][v2] = d;
			if(curBigramDoc>-1){
				// d is at least the second document with this bigram
				multiBigrams[v1][v2] = 1;
			}
		}
	}
	multiBigramIxes.resize(Doc.size());
	// create the bigram ids
	for(size_t i=0;i<Doc.size()-1;++i){
		const inttype v1 = Fea[i], v2 = Fea[i+1];
		inttype biId = 0;
		auto bi = std::make_pair(v1,v2);
		auto it = big2ix.find(bi);
		if(it==big2ix.end()){
			if(multiBigrams[v1][v2]>0){
				biId = big2ix.size()+1;
				big2ix[bi] = biId;
			}
		}else{
			biId = it->second;
		}
		multiBigramIxes[i] = biId;
	}
	COUT << big2ix.size() << " of " << (V*V) << " possible bigrams in more than one text (" << float(big2ix.size())/float(V*V) << ")" << std::endl;

	return true;
}



inline double rbeta(const double a, const double b, std::mt19937 &gen){
	const float A = std::gamma_distribution<double>(a,1.0)(gen),
		B = std::gamma_distribution<double>(b,1.0)(gen);
	return A/(A+B);
}

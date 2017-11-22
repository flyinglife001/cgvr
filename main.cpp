#include <iostream>
#include <vector>
#include <map>
#include <armadillo>
#include <sstream>
#include <fstream>
#include <string>
#include <time.h>

using namespace std;
using namespace arma;


typedef map<string,string> ss_map;

vector<double> time_list;// a list of running time
vector<double> value_list;// a list of minimun value

ss_map config;

string& trim(string& s)
{
    if(s.empty())
        return s;

    s.erase(0, s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    return s;
}

void load_config(const char* fname)
{
    ifstream in(fname);
    if(in.fail())
        cout<<"open file error! "<<fname<<endl;

    string line;
    int pos = 0;
    while(getline(in,line))
    {

        if((pos = line.find("=")) == -1)
            continue;

        string key = line.substr(0,pos);
        string value = line.substr(pos + 1);
        //trim the string
        config[trim(key)] = trim(value);
    }
    in.close();
}



int load_svmlight(const char* fname, mat& X, mat& Y, int n_dim)
{
    ifstream in(fname);
    if(in.fail())
        cout<<"open file error!"<<endl;

    string line;
    int idx;
    double value;
    char dummy;
    int max_idx = 0;
    vector<map<int,double> >  ins_list;
    vector<double> tar_list;
    while(getline(in,line))
    {
        istringstream iss(line);
        double target;
        iss>>target;
        //cout<<target<<endl;

        map<int,double> ins;
        while(iss>>idx>>dummy>>value)
        {
            //cout<<idx<<" "<<value<<endl;
            ins[idx] = value;
            if(idx > max_idx)
                max_idx = idx;
        }
        ins_list.push_back(ins);
        tar_list.push_back(target);
    }
    in.close();

    int rows = ins_list.size();
    int cols = max(max_idx,n_dim);


    X = zeros(rows,cols);
    Y = zeros(rows,1);


    map<int,double> feature;
    map<int,double>::iterator it,it_end;
    int i;

    for(i = 0; i < rows; i++)
    {
        feature = ins_list[i];
        it_end = feature.end();
        for(it = feature.begin(); it != it_end; it++)
        {
            //cout<<it->first<<" "<<it->second<<endl;
            int idx = it->first - 1;
            double value = it->second;
            X(i, idx) = value;
        }
        Y(i,0) = tar_list[i];
    }

    //assume the dataset have been scaled.
    //X = max_min_scale(X);
    //Y = max_min_scale(Y);

    //cout<<Y.t()<<endl;
    //cout<<X<<endl;

    X = join_rows(X,ones(rows,1));

    cout<<"load the data "<<X.n_rows<<"  "<<X.n_cols<<endl;

    return cols;
}

double uniform_real(double b, double a = 0)
{
    if(fabs(b - a)< 1e-6)
        return a;

    return a + (b - a)*rand()/RAND_MAX;
}


double f(mat& w, mat& X, mat& Y)
{
    string loss = config["loss"];
    double lambda = ::atof(config["lambda"].c_str());
    if(loss == "ridge")
    {
      mat diff = X*w - Y;
      int n = X.n_rows;
      //:cout<<"diff "<<diff.t()<<" ";
      mat loss = diff.t()*diff/n + lambda*w.t()*w/2;
      return loss(0,0);
    }
    else if(loss == "logistic")
    {
      /* f(x) = 1/(1 + exp(-x))  if x >= 0
       *      = exp(x)/(exp(x) + 1) if x < 0
       * */
      int i,n = X.n_rows;
      double sum = 0;
      double value;
      mat logistic = Y%(X*w);
      for(i = 0; i < n; i++)
      {
        double x = logistic(i,0);

        if( x > 0)
        {
          if(x < 20)//when x > 20, v is about 0
          {
            double v = log(1 + exp(- x));
            //cout<<x<<" "<<v<<endl;
            sum += v;
          }
        }
        else
        {
          if(x < -20)//when x < -20, v is about a linear function.
            sum += -x;
          else
          {
            value = exp(x);
            double v =  log((value + 1)/value);
            //cout<<x<<" "<<value<<" "<<v<<endl;
            sum += v;
          }
        }
      }
      mat loss = sum/n + lambda*w.t()*w/2;
      return loss(0,0);
    }
    else if(loss == "hinge" || loss == "sqhinge")
    {
      int i,n = X.n_rows;
      mat hinge = 1 - Y%(X*w);

      double sum = 0;
      for(i = 0; i < n; i++)
      {
        double v = hinge(i,0);
        if(v > 0)
        {
          if(loss == "sqhinge")
            v *= v;

          sum += v;
        }
      }

      mat loss = sum/n + lambda*w.t()*w/2;
      return loss(0,0);
    }
}

/*
mat grad_least_square(mat w, mat X, mat Y)
{
    int n = X.n_rows;
    return (2.0/n)*X.t()*(X*w - Y) + 1e-4*w;
}
*/


mat gradient(mat& w, mat& X, mat& Y)
{
    int d = X.n_cols;
    mat w_pos = zeros(d,1);
    mat w_neg = zeros(d,1);
    mat grad = zeros(d,1);
    double fabs_w,diff;
    int i,j;
    //central difference
    double epsilon = 1e-4; //it is critical

    mat e = eye(d,d);

    for(i = 0; i < d; i++)
    {
        fabs_w = fabs(w(i))*epsilon;
        w_pos = w + fabs_w*e.col(i);
        w_neg = w - fabs_w*e.col(i);

        diff = 2*fabs_w;
        grad(i) =(f(w_pos, X,Y) - f(w_neg,X,Y))/diff;
    }
    return grad;

}


mat hessian(mat& w, mat& X, mat& Y)
{
    int d = X.n_cols;
    mat w_x = zeros(d,1);
    mat w_y = zeros(d,1);
    mat w_xy = zeros(d,1);
    mat H = zeros(d,d);
    double fabs_w,diff;
    int i,j;
    //central difference
    double epsilon = 1e-4;

    mat e = eye(d,d);

    for(i = 0; i < d; i++)
    {
        for(j = 0; j < d; j++)
        {
            fabs_w = sqrt(fabs(w(i))*fabs(w(j)))*epsilon;
            w_x = w + fabs_w*e.col(i);
            w_y = w + fabs_w*e.col(j);
            w_xy = w_x + fabs_w*e.col(j);
            H(i,j) = (f(w_xy,X,Y) - f(w_x,X,Y) - f(w_y,X,Y)
                    + f(w,X,Y))/(fabs_w*fabs_w);
        }

    }
    return H;
}

//=======================================================
mat sgd(mat& X, mat& Y)
{
    time_t  t_begin = time(NULL);

    double lr = ::atof(config["lr"].c_str());
    int max_iter = ::atoi(config["max_iter"].c_str());
    int seed = ::atoi(config["seed"].c_str());
    //set the random seeds for arma
    srand(seed);

    int d = X.n_cols;
    int n = X.n_rows;
    mat w = randu(d,1);
    mat v = zeros(d,1);
    double momentum = 0.9;
    double loss;

    int i;
    mat grad;
    for(i = 0; i < max_iter; i++)
    {
       loss = f(w,X,Y);
       grad = gradient(w,X,Y);

       double grad_norm = norm(grad,2);
       cout<<"sgd epoch "<<i<<" func: "<<loss<<" grad "
           <<grad_norm<<endl;

       v = momentum*v + lr*grad;
       w = w - v;

       time_t  t_end = time(NULL);
       value_list.push_back(loss);
       time_list.push_back(difftime(t_end,t_begin));
    }

    return w;
}

//svrg: Accelerating Stochastic Gradient Descent Using Predictive
//Variance Reduction, NIPS 2013.

mat svrg(mat& X, mat& Y)
{
    time_t t_begin = time(NULL);

    int max_iter = ::atoi(config["max_iter"].c_str());
    double lr = ::atof(config["lr"].c_str());
    int seed = ::atoi(config["seed"].c_str());
    int m = ::atoi(config["m"].c_str());
    
    srand(seed);
    int n = X.n_rows, d = X.n_cols;
    mat x = randu(d);
    int k,i,j;
    mat x_tilde;
    double grad_norm;
    for(k = 0; k < max_iter; k++)
    {
        mat full_grad = gradient(x,X,Y);
        double loss = f(x,X,Y);
        x_tilde = x;
        grad_norm = norm(full_grad);

        cout<<"svrg epoch "<<k<<" func "<<loss<<" grad "<<grad_norm<<endl;
        //inner loop
        for(i = 0; i < m; i++)
        {
            int idx = (int)uniform_real(n - 1);
            mat x_row = X.row(idx);
            mat y_row = Y.row(idx);
            x -= lr*(gradient( x, x_row, y_row)
                - gradient(x_tilde, x_row, y_row )
                    + full_grad
                    );
        }


       time_t t_end = time(NULL);
       value_list.push_back(loss);
       time_list.push_back(difftime(t_end,t_begin));
    }

    return x;
}


class StepFunc
{
    private:
        mat X,Y,x,p;
    public:
        StepFunc(mat& X, mat& Y,
                mat& x, mat& p)
        {
            this->X = X;
            this->Y = Y;
            this->x = x;
            this->p = p;
        }

        double func(double alpha)
        {
            mat w = this->x + alpha*this->p;
            return f(w,this->X,this->Y);
        }

        double gradient(double alpha)
        {
            double epsilon = 1e-4;
            double fabs_w = fabs(alpha)*epsilon;
            double w_pos = alpha + fabs_w;
            double w_neg = alpha - fabs_w;

            return (this->func(w_pos) - this->func(w_neg))/(w_pos - w_neg);
        }

};




//conjugate gradient algorithm
//algorithm from line search in non-linear optimization, Jorge Nocedal and Stephen J. Wright  page. 59 - 61

double interpolation(StepFunc& f,double init_step,
    double f_zero,double grad_zero,double c1)
{
  double alpha = init_step;

  //save the two most recent value and gradient
  double last_value = 0,last_grad = grad_zero,last_f = f_zero;
  while(f.func(alpha) > f_zero + c1*alpha*grad_zero)
  {
    double grad = f.gradient(alpha);
    double f_value = f.func(alpha);
    //d1 = f'(alpha_i-1) + f'(alpha_i)
    //- 3(f(alpha_i - 1) - f(alpha_i))/(alpha_i - 1 - alpha_i)
    //d2 = sign(alpha_i - alpha_i-1)[d1^2 - f'(alpha_i-1)f'(alpha_i)]^{1/2}
    double d1 = last_grad + grad
      - 3*(last_f - f_value)/(last_value - alpha);
    double d2 = sqrt(d1*d1 - last_grad*grad);
    if(alpha < last_value)
      d2 = -d2;

    //alpha_{i+1} = alpha_i
    //- (alpha_i - alpha_i-1)(f'(alpha_i) + d2 - d1)/(f'(alpha_i)
    //- f'(alpha_i-1)+2d1)
    //
    alpha = alpha - (alpha - last_value)*(grad + d2 - d1)/(grad - last_grad + 2*d2);

    last_value = alpha;
    last_grad = grad;
    last_f = f_value;
  }
  return alpha;
}


double zoom(StepFunc& f,
        double low, double high,
        double f_zero, double grad_zero,
        double c1, double c2
        )
{
    int i;
    double alphaj;
    //cout<<"call zoom procedure"<<endl;
    for(i = 0; i < 20; i++)
    {
        alphaj = (low + high)/2.0;
        double f_alphaj = f.func(alphaj);
        if(f_alphaj > f_zero + c1*alphaj*grad_zero ||
                f_alphaj >= f.func(low))
        {
            high = alphaj;
        }
        else
        {
            double grad_alphaj = f.gradient(alphaj);
            if(fabs(grad_alphaj) <= - c2*grad_zero)
                return alphaj;

            if(grad_alphaj*(high - low) >= 0)
                high = low;

            low = alphaj;
        }
    }
    return alphaj;
}


double line_search(StepFunc& f, double alpha_max = 1)
{
    double alpha0 = 0;
    double alpha1 = uniform_real(0,alpha_max);
    double c1 = 1e-4;
    double c2 = 0.1;
    double f_zero = f.func(0);
    double grad_zero = f.gradient(0);

    int i;
    for(i = 0; i < 20; i++)
    {

       double f_alpha = f.func(alpha1);
       if(f_alpha > f_zero + c1*alpha1*grad_zero
               || (i > 0 &&  f_alpha > f.func(alpha0)))
           return zoom(f,alpha0,alpha1,f_zero,grad_zero,c1,c2);


       double grad_alpha = f.gradient(alpha1);

       if(fabs(grad_alpha) <= -c2*grad_zero)
           return alpha1;

       if(grad_alpha >= 0)
           return zoom(f,alpha1,alpha0,f_zero,grad_zero,c1,c2);

       alpha0 = alpha1;
       alpha1 = (alpha1 + alpha_max)/2;

       //cout<<"line search "<<i<<"\r";

    }

    return alpha1;
}

mat cg(mat& X, mat& Y)
{

    time_t t_begin = time(NULL);

    int max_iter = ::atoi(config["max_iter"].c_str());
    int seed = ::atoi(config["seed"].c_str());

    int n = X.n_rows, d = X.n_cols;
    srand(seed);
    mat x = randu(d);
    mat grad_value0 = gradient(x,X,Y),grad_value1;
    //cout<<grad_value0<<endl;
    mat p = - grad_value0;
    mat grad_norm = grad_value0.t()*grad_value0;
    //cout<<grad_norm<<endl;
    int k;
    double loss;
    for(k = 0; k < max_iter; k++)
    {
        loss = f(x,X,Y);
        double g_m = sqrt(grad_norm(0));

        cout<<"cg epoch "<<k<<" func "<<loss<<" grad norm "<<g_m<<endl;
        StepFunc sf(X,Y,x,p);
        double alpha = line_search(sf,1);

        x += alpha*p;

        grad_value1 = gradient(x,X,Y);


        mat beta = grad_value1.t()*(grad_value1 - grad_value0)/(grad_norm + 1e-6);

        p = - grad_value1 + max(beta(0),0.0)*p;

        grad_value0 = grad_value1;

        time_t t_end = time(NULL);
        value_list.push_back(loss);
        time_list.push_back(difftime(t_end,t_begin));

    }
    return x;
}



mat cgvr(mat& X, mat& Y)
{
  time_t  t_begin = time(NULL);
  int max_iter = ::atoi(config["max_iter"].c_str());
  int seed = ::atoi(config["seed"].c_str());
  int m = ::atoi(config["m"].c_str());
  double lr = ::atof(config["lr"].c_str());
  int L = ::atoi(config["L"].c_str());

  srand(seed);
  int n = X.n_rows, d = X.n_cols;
  mat w = randu(d);
  mat h = gradient(w, X, Y);
  mat p = -h;

  mat grad_norm = h.t()*h;
  int k,t;
  double loss;
  mat u;
  mat g0 = h, g1;
  mat x;
  int i;
  int s = (int)sqrt(n);

  for(k = 0; k < max_iter; k++)
  {
    loss = f(w,X,Y);
    double g_m = sqrt(grad_norm(0));
    cout<<"cgvr epoch "<<k<<" func "<<loss<<" grad "<<g_m<<endl;
    u = gradient(w, X, Y);
    x = w;

    p = - g0;//used by the proof.
    for(t = 0; t < m; t++)
    {
        umat idx(s,1);
        for(i = 0; i < s; i++)
        {
            idx(i,0) = (int)uniform_real(n - 1);
        }
        mat x_t = X.rows(idx);
        mat y_t = Y.rows(idx);

        StepFunc sf(x_t,y_t,x,p);
        //StepFunc sf(X,Y,x,p);
        double alpha = line_search(sf,1);

        x += alpha*p;

        if(t%L == 0)
        {
          g1 = gradient(x, x_t, y_t)
              - gradient( w, x_t, y_t)
              + u;

          grad_norm = g0.t()*g0;
          mat beta_pr = g1.t()*(g1 - g0)/(grad_norm(0) + 1e-6);
          double beta = max(beta_pr(0),0.0);
          if(beta > 10)
            beta = 0;
          p = -g1 + beta*p;
          g0 = g1;
        }
        else
        {
          x -= lr*g0;
        }
    }
    w = x;
    time_t t_end = time(NULL);
    value_list.push_back(loss);
    time_list.push_back(difftime(t_end,t_begin));
  }

  return w;
}


mat s_lbfgs_two_loop(mat g, mat S, mat& Y, int d)
{
  mat q = g;
  int m = S.n_cols;
  mat Alpha = zeros(m);
  int i;
  for(i = 0; i < m; i++)
  {
    mat s = S.col(m - i - 1);
    mat y = Y.col(m - i - 1);
    mat alpha = s.t()*q/(s.t()*y);
    q -= alpha(0)*y;
    Alpha(m - 1 - i) = alpha(0);
  }
  mat yr = Y.col(m - 1);
  mat sr = S.col(m - 1);
  mat r = sr.t()*yr/(yr.t()*yr)*q;
  for(i = 0; i < m; i++)
  {
    mat s = S.col(i);
    mat y = Y.col(i);
    mat beta = y.t()*r/(y.t()*s);
    r += (Alpha(i) - beta(0))*s;
  }
  return r;
}

mat s_lbfgs(mat& X, mat& Y)
{

  time_t  t_begin = time(NULL);

  int max_iter = ::atoi(config["max_iter"].c_str());
  double lr = ::atof(config["lr"].c_str());
  int seed = ::atoi(config["seed"].c_str());
  int m = ::atoi(config["m"].c_str());
  int L = ::atoi(config["L"].c_str());

  int n = X.n_rows, d = X.n_cols;
  int M = 10;
  //int b = 100;
  int b = (int)sqrt(n);
  int bH = b;

  srand(seed);
  mat w = randu(d);
  mat last_ur = zeros(d);
  mat slist,ylist;

  int r = 0, k;
  mat uk;
  double loss;
  mat Hv,p,x;
  int i,t;
  for(k = 0; k < max_iter; k++)
  {
    uk = gradient(w, X, Y);
    x = w;
    loss = f(w, X, Y);
    double grad_norm = norm(uk,2);
    cout<<"s-lbfgs epoch "<<k<<" func "<<loss<<" grad "<<grad_norm<<endl;
    mat xlist;
    for(t = 0; t < m; t++)
    {
       umat s_idx(b,1);
       for(i = 0; i < b; i++)
       {
         s_idx(i,0) = (int)uniform_real(n - 1);
       }
       mat x_t = X.rows(s_idx);
       mat y_t = Y.rows(s_idx);
       mat v = gradient(x, x_t, y_t) - gradient(w, x_t, y_t) + uk;

       xlist.insert_cols(xlist.n_cols, x);
       if(xlist.n_cols > L)
         xlist.shed_cols(0,0);

       if(r < M)
         p = v;
       else
         p = Hv;

       x -= lr*p;
       //cout<<p.t()<<endl;

       if(t%L == 0 &&  r > 0)
       {
         r += 1;
         mat ur = mean(xlist,1);
         //cout<<ur<<endl;

         umat t_idx(bH,1);
         for(i = 0; i < bH; i++)
         {
           t_idx(i,0) = (int)uniform_real(n - 1);
         }

         mat x_t = X.rows(t_idx);
         mat y_t = Y.rows(t_idx);

         mat sr = ur - last_ur;
         mat yr = hessian(ur, x_t, y_t)*sr;

         if(r >= M)
         {
           Hv = s_lbfgs_two_loop(v,slist,ylist,d);
           slist.shed_cols(0,0);
           ylist.shed_cols(0,0);
         }

         slist.insert_cols(slist.n_cols, sr);
         ylist.insert_cols(ylist.n_cols, yr);

         last_ur = ur;
       }
    }
    w = x;
    time_t  t_end = time(NULL);
    value_list.push_back(loss);
    time_list.push_back(difftime(t_end,t_begin));
  }
  return w;
}

mat predict(mat& X, mat& w)
{
  return X*w;
}


double estimate_acc(mat& pred, mat& Y)
{
  int i;
  int n = Y.n_rows;
  double accuracy = 0.0;
  for(i = 0; i < n; i++)
  {
    //cout<<Y(i)<<" "<<pred(i)<<endl;

    if(Y(i)*pred(i) >= 0)
      accuracy += 1;
  }
  //cout<<"acc: "<<accuracy<<" n: "<<n<<endl;
  return accuracy/n;
}

/* ***********************************************************************
 *
 *
 * N P N P P
 * score  0.2 0.3 0.4 0.5 0.6
 * ri 1 2 3 4 5
 *
 * [sum r_i - (P + 1)P/2]/(P*N)
 * */
double estimate_auc(mat& pred, mat& Y)
{
  int i;
  int n = Y.n_rows;
  double auc = 0.0;
  uvec idx = sort_index(pred);
  Y = Y.rows(idx);
  pred = pred.rows(idx);
  //cout<<Y.t()<<endl;

  double last_v = -1e8, v;
  int b_idx = 0, e_idx = 0,k;
  vec ranks(n);
  for(i = 0; i < n; i++)
  {
    ranks(i) = i + 1;
    v = pred(i);
    if(fabs(v - last_v) > 1e-6)
    {
      e_idx = i;
      double base_value = ranks(b_idx);
      for(k = b_idx; k < e_idx; k++)
        ranks(k) = base_value + (e_idx - b_idx - 1)/2.0;

      b_idx = e_idx;
    }
    last_v = v;
  }
  //process the last range
  e_idx = n;
  double base_value = ranks(b_idx);
  for(k = b_idx; k < e_idx; k++)
    ranks(k) = base_value + (e_idx - b_idx - 1)/2.0;

  //cout<<ranks<<endl;

  double sum_pos_ranks = 0, n_pos = 0;
  for(i = 0; i < n; i++)
  {
    int label = (int)Y(n - 1 - i);
    if(label == 1)
    {
      sum_pos_ranks += ranks(n - 1 - i);
      n_pos += 1;
    }
  }
  //cout<<"sum "<<sum_pos_ranks<<" "<<n_pos<<endl;

  if(n_pos == 0 || n_pos == n)
    return 1;

  return (sum_pos_ranks - (n_pos + 1.0)*n_pos/2)/(n_pos*(n - n_pos));

}

int tmain(int argc, char** argv)
{
  //mat y("0 0 1 1");
  //mat y("1 0 1 0");
  //mat pred("0.1, 0.4, 0.35, 0.8");
  //mat pred("0.2 0.3, 0.6, 0.4");
  //y = y.t();
  //pred = pred.t();
  //cout<<estimate_auc(pred,y)<<endl;
  mat X,Y;
  int n_dim = 0;
  load_svmlight("/home/vision/data/cgvr/a1a",X,Y,0);
  cout<<X.row(0)<<endl;

}


int main(int argc, char** argv)
{
    /*ss_map cfg;
    const char* cfg_file = "/home/vision/cppfile/cgvr/input.cfg";
    load_config(cfg_file,cfg);
    cout<<cfg.size()<<endl;
    ss_map::iterator it, it_end = cfg.end();
    for(it = cfg.begin(); it != it_end; it++)
    {
        cout<<it->first<<" "<<it->second<<endl;
    }*/

    if(argc < 2)
    {
      cout<<"no assigned options ..."<<endl;
      exit(-1);
    }

    int i;
    for(i = 1; i < argc; i += 2)
    {
      string key = (string)argv[i];
      string value = (string)argv[i + 1];
      config[key] = value;
    }

    string task = config["task"];
    if(task == "converge")
    {
      mat X,Y;
      string fname = "/home/vision/data/cgvr/" + config["dt"];
      int cols = load_svmlight(fname.c_str(),X,Y,0);
      if(config["name"] == "cg")
        cg(X,Y);
      else if(config["name"] == "cgvr")
        cgvr(X,Y);
      else if(config["name"] == "sgd")
        sgd(X,Y);
      else if(config["name"] == "s_lbfgs")
        s_lbfgs(X,Y);
      else if(config["name"] == "svrg")
        svrg(X,Y);

      //write the json format file.
      cout<<"{";

      ss_map::iterator it, it_end = config.end();
      for(it = config.begin(); it != it_end; it++)
      {
        cout<<"\""<<it->first<<"\":"<<"\""<<it->second<<"\",";
      }

      cout <<"\"value\":[";
      int len = value_list.size();
      for(i = 0; i < len - 1; i++)
        cout<<value_list[i]<<",";
      cout<<value_list.back()<<"],";
      cout<<"\"time\":[";
      for(i = 0; i < len - 1; i++)
        cout<<time_list[i]<<",";
      cout<<time_list.back()<<"]}"<<endl;
    }
    else//classification task.
    {
      int dim = 0;
      if(config.find("dim") != config.end())
        dim = ::atoi(config["dim"].c_str());

      mat trainX, trainY;
      mat validX, validY;
      mat testX, testY;

      string fname = "/home/vision/data/cgvr/" + config["dt"] + ".r";
      int cols = load_svmlight(fname.c_str(),trainX,trainY,dim);
      fname = "/home/vision/data/cgvr/" + config["dt"] + ".v";
      load_svmlight(fname.c_str(),validX,validY,cols);
      fname = "/home/vision/data/cgvr/" + config["dt"] + ".t";
      load_svmlight(fname.c_str(),testX,testY, cols);

      mat w;
      if(config["name"] == "cg")
        w = cg(trainX,trainY);
      else if(config["name"] == "cgvr")
        w = cgvr(trainX,trainY);
      else if(config["name"] == "sgd")
        w = sgd(trainX,trainY);
      else if(config["name"] == "s_lbfgs")
        w = s_lbfgs(trainX,trainY);
      else if(config["name"] == "svrg")
        w = svrg(trainX,trainY);

      //cout<<w.t()<<endl;
      mat pred = predict(validX,w);
      double valid = estimate_auc(pred,validY);

      pred = predict(testX,w);
      double test = estimate_auc(pred,testY);

      //write the json format file.
      cout<<"{";

      ss_map::iterator it, it_end = config.end();
      for(it = config.begin(); it != it_end; it++)
      {
        cout<<"\""<<it->first<<"\":"<<"\""<<it->second<<"\",";
      }

      cout<<"\"valid\":"<<valid<<","
        <<"\"test\":"<<test<<",";

      cout <<"\"value\":[";
      int len = value_list.size();
      for(i = 0; i < len - 1; i++)
        cout<<value_list[i]<<",";
      cout<<value_list.back()<<"],";
      cout<<"\"time\":[";
      for(i = 0; i < len - 1; i++)
        cout<<time_list[i]<<",";
      cout<<time_list.back()<<"]}"<<endl;

    }




}

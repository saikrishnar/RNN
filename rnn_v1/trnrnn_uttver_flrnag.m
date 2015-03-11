for NE = 1:nepochs
    
    iter = (NE-1)*train_numbats;
    valerr = 0;
    lgWi = 0;
    lgW = 0;
    
    for j = 1:train_numbats
        
        
        sl = train_clv(j+1) - train_clv(j);
        
        % nag equations
        GWi1 = GWi + mf*GpdWi; GW1 = GW + mf*GpdW; GU1 = GU + mf*GpdU;
        Gbh1 = Gbh + mf*Gpdbh; Gbo1 = Gbo + mf*Gpdbo;
        
        mt = []; X = train_batchdata(train_clv(j):train_clv(j+1)-1,:);
        [hcm,ym] = fp_rnn(X,GWi1,GW1,GU1,Gbh1,Gbo1,ff,Nh,sl);
        mt = train_batchtargets(train_clv(j):train_clv(j+1)-1,:);

        % bacward prop
        % Compute deltas of output layer weights and biases
        delo = (ym-mt);
        gU = (delo'*hcm)/sl;
        gbo = mean(delo)';
        
        
        % Compute gradients of recurrent weights and biases
        delnt = gpuArray(zeros(Nh,1));
        delm = gpuArray(zeros(sl,Nh));
        
        iemat = GU1'*delo';
        GW1 = GW1';
        
        % backward prop
        derf = (bby2a*(a_tanh_sqr - hcm.^2))';
        for k = sl:-1:1
            ie = derf(:,k).*(GW1*delnt+iemat(:,k));
            delnt = ie;
            delm(k,:) = delnt;
        end
        
        hpm = [h_0';hcm(1:end-1,:)];
        gW = (delm'*hpm)/sl;
        gbh = mean(delm)';
        
        % Compute gradients of inpu-hidden layer weights
        gWi = (delm'*X)/sl;
        
        lgWi = lgWi + sqrt(sum(gWi.^2,2))/train_numbats;
        lgW = lgW + sqrt(sum(gW.^2,2))/train_numbats;
        
        dWi = -eta*gWi;
        dW = -eta*gW;
        dU = -eta*gU;
        dbh = -eta*gbh;
        dbo = -eta*gbo;
               
        GpdU = dU + mf*GpdU;
        Gpdbo = dbo + mf*Gpdbo;
        GpdW = dW + mf*GpdW;
        Gpdbh = dbh + mf*Gpdbh;
        GpdWi = dWi + mf*GpdWi;
        
        % Update params
        GU = GU + (GpdU) ;
        Gbo = Gbo + (Gpdbo) ;
        GW = GW + GpdW;
        Gbh = Gbh + Gpdbh ;
        GWi = GWi + (GpdWi) ;
  
       
        GW = GW.*mask;
        GW = GW*(1.2/abs(eigs(double(gather(GW)),1,'lm',opts)));
        
        num_up = num_up + 1;
        
        if mod(num_up,25) == 0
            
            tic
            valerr = 0;
            % compute error on validation set
            for li = 1:(val_numbats)
                mt = []; X = [];
                sl = val_clv(li+1) - val_clv(li);
                X = val_batchdata(val_clv(li):val_clv(li+1)-1,:);
                [hcm,ym] = fp_rnn(X,GWi,GW,GU,Gbh,Gbo,ff,Nh,sl);
                
                mt = val_batchtargets(val_clv(li):val_clv(li+1)-1,:);

                [me] = mean(sum((mt - ym).^2,2)./(sum(mt.^2,2)));
                valerr = valerr + me/(val_numbats);
            end
            toc
            
            %     Print error (validation) per epoc
            fprintf('Update : %d  Val Loss : %f \n',num_up,valerr);
            
            tic
            if valerr < best_val_loss
                if valerr < (best_val_loss*imp_thresh)
                    patience = max(patience,iter*patience_inc);
                end
                best_val_loss = valerr;
                best_iter = iter;
                
                testerr = 0;
                % compute error on test set
                for li = 1:(test_numbats)
                    mt = [];
                    sl = test_clv(li+1) - test_clv(li);
		    X = test_batchdata(test_clv(li):test_clv(li+1)-1,:);	
                    [hcm,ym] = fp_rnn(X,GWi,GW,GU,Gbh,Gbo,ff,Nh,sl);
                   
                    mt = test_batchtargets(test_clv(li):test_clv(li+1)-1,:);

                    me = mean(sum((mt - ym).^2,2)./(sum(mt.^2,2)));
                    testerr = testerr + me/(test_numbats);
                end
                %toc(ttde)
                
                % Print error (validation and testing) per epoc
                fprintf(fid,'%d %d %f %f \n',NE,num_up,valerr,testerr);
                
                % Print error (testing) per epoc
                fprintf('\t Update : %d  Test Loss : %f \n',num_up,testerr);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%% save weight file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % save parameters every epoch
                Wi = gather(GWi);W = gather(GW);U = gather(GU); bh = gather(Gbh); bo = gather(Gbo);
                save(strcat(wtdir,'W_',arch_name,'.mat'),'Wi','W','U','bh','bo');
            end
            toc
            
        end
        
        if isnan(valerr)
            break;
        end
        
        if num_up > 900;
            mf = mf_max;
        end
        
        if num_up > 4500;
            mf = 0.9;
        end
        
    end
    fprintf('avg gradients : %f %f\n',mean(lgWi),mean(lgW));
    
    if isnan(valerr)
        break;
    end
    
end

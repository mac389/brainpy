						cprint('\tClustering','red')
						clusters = [tech.cluster(transpose(PC[1])) for PC in PCs]  #<------Notice the [1] index to retrive just the projections
						cprint('\tClustered','green')
						'''My adpatation of Pycluster uses the silhouette coefficient to find a good number of clusters. 
						
						   Output of tech.cluster for EACH snippet-matrix
						   
						   {'clustermap': clustermap,
				  			'centroids':centroids,
							'distances':array(m),
							'mass':mass,
							'silhouettes':sil_co}
							
							To choose the best number of clusters, choose the dictionary with the highest sil_co
						'''		
						cPickle.dump([PCs,clusters],open(basepath+'/'+filename+'shank%d_channel%d.pkl'%(shank,channel),'wb'))
						
						clusters = [sorted(cluster,key=lambda attempt: attempt['silhouettes'])[-1] for cluster in clusters]
						#This arranges the clustering attempts for each snippet-matrix in order of ascending silhouette coefficients.
				
						color = ['r','k','g','b','m','DarkOrange','purple']
						for k,cluster in enumerate(clusters):
							fig = figure(facecolor='white')
							hold(True)
							
							PC = PCs[k][1] #<-- Remember from above how [1] choses the projections of the data
							
							ul = subplot2grid((8,4),(0,0),colspan=2,rowspan=4)
							ll=subplot2grid((8,4),(4,0),colspan=2,rowspan=4)
							lr=subplot2grid((8,4),(4,2),colspan=2,rowspan=4)
							
							wf_panels = [subplot2grid((8,4),(m%4,2+m/4)) for m in range(len(channel_names))]
							
							for n in range(max(cluster['clustermap'])):
								hold(True)		
								
								PC1 = PC[0,cluster['clustermap']==n]
								PC2 = PC[1,cluster['clustermap']==n]
								PC3 = PC[2,cluster['clustermap']==n]
								
								#Upper Left
								ul.scatter(PC2,PC1,c=color[n])
								hold(True)
								
								#Upper Right-waveform panel
								current_wfs = waveforms[k][cluster['clustermap']==n,:]
								wfs = average(current_wfs,axis=0)
								wfs = reshape(wfs,(cut_duration,-1))
								for p,panel in enumerate(wf_panels):
									panel.plot(wfs[:,p],color[n])
									panel.axes.get_yaxis().set_visible(False)
									panel.axes.get_xaxis().set_visible(False)
									hold(True)
							 			
							 	#Lower Left
								ll.scatter(PC3,PC1,c=color[n])
								
								#Lower Right
								lr.scatter(PC3,PC2,c=color[n])
							
							for panel in [ll,lr]:
								postdoc.adjust_spines(panel,['bottom','left'])
							
							postdoc.adjust_spines(ul,['left'])
							hold(True)	
							'''
								Figure laid out as below. It is the lower triangle of a 3 X 3 matrix, not including the diagonal.
								
								
								   |				  |
							 PC2   |		  Volts   |		 <--- The upper right panel show the waveforms
								   |_______		  	  |________
									  PC1				Time
								
								   |				  |
							 PC3   |		   PC3	  | 
								   |________		  |________
									  PC1				  PC2
							'''
							savefig(basepath+'/shank%d_channel%d.png'%(shank,channel),dpi=150)
							print 'Saved spiketimes for channel %d' % channel
							
							#-- Clear the screen
							print chr(27) + '[2J'
							#-------------------
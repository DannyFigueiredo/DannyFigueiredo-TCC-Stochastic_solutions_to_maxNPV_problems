steepestAsc <- function(){
	library(triangle)

	#==============#
	#  Parametros  #
	#==============#
	Ns = 100					# numero de simulacoes
	n = 14					# numeros de atividades de um projeto
	dMax = 44					# prazo final para entrega do projeto (deadline)
	fee = 0.01					# taxa de desvalorizacao do dinheiro
	# vetor de duracao das atividades
	d = c(0,6,5,3,1,6,2,1,4,3,2,3,5,0)	
	# lista de atividades sucessoras
	suc = list( c(2,3,4),9,c(5,6,7),8,10,12,c(8,11),13,14,12,12,13,14,0 )
	# lista de atividades predecessoras
	pre = list( 0,1,1,1,3,3,3,c(4,7),2,5,7,c(6,10,11),c(8,12),c(9,13) )
	# vetor de valores do fluxo de caixa de cada atividade
	vals = c(0,-140,318,312,-329,153,193,361,24,33,387,-386,171,0)
	# vetor de resposta da simulacoes de NPV
	resNPV = vector( mode = "numeric", length = Ns )
	# matriz de agendamento de atividade em cada amostragem
	resSched = matrix( ncol = n, nrow = Ns )

	# vetores de retorno de agendamento mais cedo e mais tarde
	est <- vector( mode = "numeric", length = n ) # tempo mais cedo da atividade
	eft <- vector( mode = "numeric", length = n ) # tempo mais tarde
	lst <- vector( mode = "numeric", length = n ) # tempo mais tarde que pode terminar
	lft <- vector( mode = "numeric", length = n )

	#===========#
	#  Funcoes  #
	#===========#
	evalNPV = function( sched ){
		# Valor Presente Liquido para agendamento est (mais cedo)
		t = sched + d
		npv = sum( vals * exp( -fee*t ))
		npv
	}

	val = function( sched ){
		# verifica se o agendamento eh valido
		val = TRUE
		for ( i in 2:n ){
			pr = pre[[i]]
			for ( p in pr ){
				if ( sched[i] < sched[p]+d[p] ){
					val = FALSE
					break()
				}
			}
		}
		val
	}

	cpmf = function ( s, est, eft ){
		# encontra o agendamento MAIS CEDO para iniciar a atividade e 
		# o agendamento MAIS CEDO para o final da atividade
		eft[s] = est[s] + d[s]
		if ( suc[[s]][1] != 0 ){
			for ( i in suc[[s]] ){
				if ( est[i] < eft[s] ){
					est[i] = eft[s]
				}
				est = cpmf( i, est, eft )
			}
		}
		est
	}

	cpmb = function (s, lft, lst){
		# encontra o agendamento MAIS TARDE para iniciar e
		# terminar cada atividade
		lst[s] = lft[s] - d[s]
		if ( pre[[s]][1] != 0 ){
			for ( i in pre[[s]] ){
				if( lft[i] > lst[s] ){
					lft[i] = lst[s]
				}
				lft = cpmb( i, lft, lst ) 
			}
		}
		lft
	}

	cpmDriver = function( i = 1, est, eft ){
		est = cpmf(1, est, lst)			# chama o passo crescente
		eft = est + d
		lft = rep( dMax, times = n )		# vetor de deadline
		lft = cpmb( n, lft, lst )		# chama o passo de retorcesso
		r = list( est = est, lft = lft )
		r
	}

	adjEst = function( s, est ){
		if ( suc[[s]][1] != 0 ){
			for ( i in suc[[s]] ){
				if ( est[i] < ( est[s] + d[s]) ){
					est[i] = est[s] + d[s]
				}
				est = adjEst( i, est )
			}
		}
		est
	}

	findMaxGradientNeighbour = function ( sched ){
		maxNPV = evalNPV( sched )
		maxSched = sched

		# encontrando o no de maior gradiente
		for ( node in 2:(n-1) ){
			# 1 - parametros do no
			tsched = sched
			max = lst[node]
			
			# 2 - aumenta o agendamento do no
			if ( tsched[node] < max ){
				tsched[node] = tsched[node] + 1
			}
			
			# 3 - gera um novo agendamento
			tsched = adjEst( node, tsched )
			
			# 4 - gera novo NPV para o agendamento
			val = evalNPV( tsched )

			# 5 - compara
			if ( val > maxNPV ){
				maxSched = tsched
				maxNPV = val
			}
		}
		r = maxSched
		r	
	}

	sa = function ( sched ){
		seqNPV = vector( mode = "numeric", length = Ns)
		seqSched = matrix( 0, nrow = Ns, ncol = n )
		seqSched[1, ] = sched
		t = 0
		
		repeat {
			# 1 - encontrar o agendamento vizinho
			t = t + 1
			nSched = findMaxGradientNeighbour( sched )
			if ( val( nSched ) ){
				# seqSched[t, ] = nSched

				# 2 - valorizar NPVs
				npv = evalNPV( sched )
				seqNPV[t] = npv
				nnpv = evalNPV( nSched )
				delta = nnpv - npv

				# 3 - verifica aceitação
				if ( delta > 0 ){
					sched = nSched
					seqSched[t, ] = nSched
					seqNPV[t] = nnpv
				} else {
					seqSched = seqSched[1:t, ]
					seqNPV = seqNPV[1:t]
					break
				}
			} else {
				# agendamento invalido
				print( c("Agendamento invalido", nSched) )
			}
		}
		r = list( npv = seqNPV, seq = seqSched )
		r
	}

	#========#
	#  Main  #
	#========#
	# Preparacao dos dados
	r1 = cpmDriver (1, est)
	est = r1$est
	eft = est + d
	lft = r1$lft
	lst = lft - d

	# Chama a função de Steepest Ascent
	r2 = sa( est )
	
	# Representacao grafica
	plot( r2$npv )

	print( tail( r2$seq ) )
	r2
}
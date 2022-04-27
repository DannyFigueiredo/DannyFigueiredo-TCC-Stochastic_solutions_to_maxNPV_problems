simAnnealing = function(){
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

	# vetores de retorno de agendamento mais cedo e mais tarde
	est <- vector( mode = "numeric", length = n ) # tempo mais cedo de inicio da atividade
	eft <- vector( mode = "numeric", length = n ) # tempo mais cedo de final da atividade
	lst <- vector( mode = "numeric", length = n ) # tempo mais tarde para iniciar a atividade
	lft <- vector( mode = "numeric", length = n ) # tempo mais tarde para finalizar a atividade

	#===========#
	#  Funcoes  #
	#===========#
	evalNPV = function( sched ){
		# Valor presente liquido para um agendamento para inicio mais cedo da atividade
		t = sched + d
		npv = sum( vals * exp( -fee*t ) )
		r = npv
	}

	val = function( sched ){
		val = TRUE
		for ( i in 2:n ){
			pr = pre[[i]]
			for ( p in pr ){
				if ( sched[i] < sched[p] + d[p] ){
					val = FALSE
					break()
				}
			}
		}
		val
	}

	cpmf = function( s, est, eft ){
		# encontra o inicio e final mais cedo das atividades
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

	cpmb = function( s, lft, lst ){
		# encontraa o inicio e final mais tarde para as atividades
		lst[s] = lft[s] - d[s]
		if ( pre[[s]][1] != 0 ){
			for ( i in pre[[s]] ){
				if ( lft[i] > lst[s] ){
					lft[i] = lst[s]
				}
				lft = cpmb( i, lft, lst )
			}
		}
		lft
	}

	adjest = function( s, est ){
		if ( suc[[s]][1] != 0 ){
			for ( i in suc[[s]] ){
				if ( est[i] < (est[s] + d[s]) ){
					est[i] = est[s] + d[s]
				}
				est = adjest( i, est )
			}
		}
		est
	}

	cpmDriver = function ( i = 1, est, eft ){
		est = cpmf( 1, est, lst )
		eft = est + d
		lft = rep( dMax, times = n)
		lft = cpmb( n, lft, lst )
		r = list( est = est, lft = lft)
		r
	}

	findNeighbourSched = function( sched ){
		# amostra
		node = sample( 1:n, 1, replace = TRUE )
		# print( node )

		# parametro do no
		min = sched[node]
		max = lst[node]

		if ( sched[node] < max ){
			t = sched[node] + 1
			# t = sample( min:max, 1 )
			sched[node] = t
		}
	
		# avalia o novo agendamento de iniciação mais cedo
		sched = adjest( node, sched )

		# retorno
		r = sched
		r
	}

	sa = function ( sched ) {
		seqNPV = vector( mode = "numeric", length = Ns )
		seqSched = matrix( 0, nrow = Ns, ncol = n )
		seqSched[1, ] = sched
		seqP = vector( mode = "numeric", length = Ns )

		for ( t in Ns:2 ){
			# 1 - encontrar o agendamento vizinho
			nSched = findNeighbourShed( sched )
			seqSched[T, ] = nSched

			# 2 - avaliar NPVs
			npv = evalNPV( sched )
			seqNPV[t] = npv
			nnpv = evalNPV( nSched )
			delta = nnpv - npv

			# 3 - verifica aceitacao
			if( delta > 0 ){
				sched = nSched
				seqNPV[t] = nnpv
				est = sched
			} else {
				s = ( 10/ log(t, 10) )
				p = min( 1, exp(delta*s) )
				seqP[t] = p
				u = rbinom(1, 1, p)
				if ( u == 1 ){
					sched = nSched
					seqNPV[t] = nnpv
				}
			}
		}
		
		seqNPV = seqNPV[Ns:1]
		seqP = seqP[Ns:1]
		r = list( npv = seqNPV, seq = seqSched, p = seqP )
		r
	}

	#========#
	#  Main  #
	#========#
	# 1 - Preparacao dos dados
	r1 = cpmDriver( 1, est )
	est = r1$est
	eft = est + d
	lft = r1$lft
	lst = lft - d

	# 2 - chama Simulated Annealing
	r2 = sa( est )
	plot( r2$p )
	r2
}
int read_input_list(std::vector<float> *values, char **args, int argi, int nargs, bool debug = false)
{
	if (debug) printf("Reading list \n");
		
	int go = 1;
	float val0, lastval;
	lastval = -1;
	//printf("i0 = %d added\n", i0);
	std::string link;
	
	while(go == 1)
	{
		val0 = atof(args[argi]);
		argi ++;
		
		if(debug) printf(" val0 = %.2f ", val0);	
		if(val0 != lastval) values -> push_back(val0);
		lastval = val0;
		
		if(!(argi < nargs))
			return argi;
		
		link = std::string(args[argi]);
		argi++;

		if(link == "and")
		{
			go = 1;
		}
		else if (link == ":")
		{
			if (debug) printf(":");
			float d = 1.0;			
			float valend = atof(args[argi]);
			argi ++;
			
			if(argi < nargs)
			{
				// check if there is another : to form a : d : b
				link = std::string(args[argi]);
				argi ++;
				
				if (link == ":" && argi < nargs)
				{
					d = valend;
					valend = atof(args[argi]);
					argi ++;
				}
				else
				{
					argi --;		
				}
			}
			if(debug) printf(" d = %.2f end = %.2f", d, valend);
			// Add value. i0 was already added
			for(float val = val0 + d; val <= valend + 0.0000001 ; val += d)
			{
				lastval = val;
				values -> push_back(val);
			}
			
			if(argi < nargs)
			{
			
				link = std::string(args[argi]);
				argi++;
				if(link == "and")
				{
					go = 1;
				}
				else
				{
					go = 0;
					argi --;
				}
			}
			else
			{
				return argi;
			}
			
		}
		else
		{
			go = 0;
			argi--;
		}
		
	}
	if(debug) printf("\n");
	return argi;

}


int read_input_list(std::vector<int> *values, char **args, int argi, int nargs, bool debug = false)
{
	if(debug) printf("Reading Integer list \n");
		
	int go = 1;
	int val0, lastval;
	lastval = -1;
	//printf("i0 = %d added\n", i0);
	std::string link;
	
	while(go == 1)
	{
		val0 = atoi(args[argi]);
		argi ++;
		
		if(val0 != lastval) values -> push_back(val0);
		lastval = val0;
		
		if(debug) printf("val %d added\n", val0);
		
		if(!(argi < nargs))
			return argi;
			
		link = std::string(args[argi]);
		argi++;

		if(link == "and")
		{
			go = 1;
		}
		else if (link == ":")
		{
			printf(":");
			int d = 1;			
			int valend = atoi(args[argi]);
			argi ++;
			
			if(argi < nargs)
			{
				// check if there is another : to form a : d : b
				link = std::string(args[argi]);
				argi ++;
				
				if (link == ":" && argi < nargs)
				{
					d = valend;
					valend = atoi(args[argi]);
					argi ++;
				}
				else
				{
					argi --;		
				}
			}
			// Add value. i0 was already added
			for(int val = val0 + d; val < valend+1 ; val += d)
			{
				lastval = val;
				values -> push_back(val);
				if(debug) printf("val %d added\n", val);
			}
			
			if(argi < nargs)
			{			
				link = std::string(args[argi]);
				argi++;
				if(link == "and")
				{
					go = 1;
				}
				else
				{
					go = 0;
					argi --;
				}
			}
			else
			{
				return argi;
			}
			
		}
		else
		{
			go = 0;
			argi--;
		}
		
	}
	if(debug) printf("\n");
	return argi;

}









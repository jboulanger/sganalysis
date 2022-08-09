#@File(label="input file",description="Select the cells.csv file generated [figure] mode") path_cells_csv

/*
 * Pivot Table
 * 
 * Convert a long table to a wide table.
 * 
 * Jerome Boulanger for Yaiza and Lucas
 * 
 */
 
 
// open the table
Table.open(path_cells_csv);
// get the list of the headings
headings = split(Table.headings,"\t");
// ask for the column/value variables
Dialog.create("Select variables");
Dialog.addChoice("Column", headings);
Dialog.addChoice("Value", headings);
Dialog.show();
col = Dialog.getChoice();
val = Dialog.getChoice();

// get the unique values of column
columns = Table.getColumn(col);
values = Table.getColumn(val);

// create the pivot table
Table.create(val);
pivotTable(columns,values);

function pivotTable(columns,values) {
	// pivot table
	// input columns and values are array with value from the original table
	if (columns.length != values.length) {
		exit("pivotTable: columns and values should have the same length");		
	}
	headers = arrayUnique(columns);
	rows = newArray(headers.length);	
	for (i = 0; i < values.length; i++) {
		k = arrayFirstIndexOf(headers, columns[i]);
		if (k >= 0) {
			Table.set(headers[k], rows[k], values[i]);
			rows[k] = rows[k] + 1;
		}
	}
	// fill the shorter columns with NaNs
	n = Table.size;
	for (k = 0; k < headers.length; k++) {
		for (i = rows[k]; i < n; i++) {
			Table.set(headers[k], i, "NaN");
		}
	}
}

function arrayFirstIndexOf(A,x) {	
	// first index of A where A[i] == x
	for (i = 0; i < A.length; i++) {
		if ((A[i]==x)) {
			return i;
		}
	}
	return -1;
}

function arrayUnique(A) {
	// return the list of unique elements of the array A
	B = A;	
	N = 1;
	for (i = 1 ; i < A.length; i++) {
		found = false;
		for (j = 0 ; j < N; j++) {
			if ((A[i]==B[j])) {
				found = true;
				break;
			}
		}
		if (!found) {			
			N++;
			B[N-1] = A[i];
		}
	}
	return Array.trim(B,N);
}
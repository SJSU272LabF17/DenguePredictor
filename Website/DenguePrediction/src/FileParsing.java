import java.io.BufferedReader;
import java.io.FileReader;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.ArrayList;

public class FileParsing {

	private static final String COMMA_DELIMITER = ",";

	class Detail {
		String city;
		int year;
		int week;
		int cases;

		public String getCity() {
			return city;
		}

		public void setCity(String city) {
			this.city = city;
		}

		public int getYear() {
			return year;
		}

		public void setYear(int year) {
			this.year = year;
		}

		public int getWeek() {
			return week;
		}

		public void setWeek(int week) {
			this.week = week;
		}

		public int getCases() {
			return cases;
		}

		public void setCases(int cases) {
			this.cases = cases;
		}
	}

	public void readFile(String strFilePath) {
		BufferedReader br = null;
		try {
			// Reading the csv file
			br = new BufferedReader(new FileReader(strFilePath));
			String line = "";
			ArrayList<Detail> detailsLst = new ArrayList<Detail>();
			// Read to skip the header
			br.readLine();
			// Reading from the second line
			while ((line = br.readLine()) != null) {
				String[] details = line.split(COMMA_DELIMITER);
				if (details.length > 0) {
					// Save the employee details in Employee object
					Detail detail = new Detail();
					detail.setCity(details[0]);
					detail.setYear(Integer.parseInt(details[1]));
					detail.setWeek(Integer.parseInt(details[2]));
					detail.setCases(Integer.parseInt(details[3]));
					detailsLst.add(detail);
				}
			}
			insertData(detailsLst);
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	
	public void insertData(ArrayList<Detail> data){

		try {
			Class.forName("com.mysql.jdbc.Driver");
			Connection con = DriverManager.getConnection(
					"jdbc:mysql://localhost:3306/dengueprediction", "root", "root");

			PreparedStatement oPrStmt = con
					.prepareStatement("INSERT INTO PredictionData (city,year,week,cases) VALUES(?,?,?,?)");
				
			int nLength = data.size();
			Detail detail = null;
			while(nLength>0){
				detail =  data.get(--nLength);
				int nIndex = 0;
				oPrStmt.setString(++nIndex, detail.getCity());// parameter index start from 1
				oPrStmt.setInt(++nIndex, detail.getYear());
				oPrStmt.setInt(++nIndex, detail.getWeek());
				oPrStmt.setInt(++nIndex, detail.getCases());
				oPrStmt.addBatch();
			}
			oPrStmt.executeBatch();
			
			oPrStmt.close();
			con.close();
		} catch (Exception e) {
			System.out.println(e);
		}
	}
}
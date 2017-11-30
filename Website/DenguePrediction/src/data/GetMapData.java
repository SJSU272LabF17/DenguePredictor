package data;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.HashMap;

import org.json.JSONArray;
import org.json.JSONObject;


public class GetMapData {
	
	
	public JSONArray getDataArray(String year,int week){

		//StringBuilder sbData = new StringBuilder("[");
		JSONArray oJsonArr = new JSONArray();
		try {
			Class.forName("com.mysql.jdbc.Driver");
			Connection con = DriverManager.getConnection(
					"jdbc:mysql://localhost:3306/dengueprediction", "root", "root");

			StringBuilder sbSql = new StringBuilder("SELECT SUM(cases) as cases,c.lng as lng, c.lat as lat FROM PredictionData pd JOIN ")
								.append(" cities c ON c.cityname = pd.city ")
								.append(" WHERE pd.YEAR= ? ");
			if(week>0){
				sbSql.append(" AND pd.week = ?");
			}
			sbSql.append(" GROUP BY c.lng,c.lat " );
			PreparedStatement oPrStmt = con
					.prepareStatement(sbSql.toString());
			
				
				int nIndex = 0;
				oPrStmt.setString(++nIndex, year);
				if(week>0){
					oPrStmt.setInt(++nIndex, week);
				}
				System.out.println(oPrStmt.toString());	
			ResultSet oRs =  oPrStmt.executeQuery();
			
			while(oRs.next()){
				System.out.println(oRs.getString("lat")+" "+oRs.getString("lng") );
				int nCount =  oRs.getInt("cases");
				while(--nCount>=0){
					JSONObject obj = new JSONObject();
					obj.put("lat", oRs.getString("lat"));
					obj.put("lng", oRs.getString("lng"));
					oJsonArr.put(obj);
				}
			}
			oRs.close();
			oPrStmt.close();
			con.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e);
		}
		//sbData.append("]");
		System.out.println(oJsonArr.toString());
	return oJsonArr;
	}
	
	public ArrayList<Integer> getYearData(){
		ArrayList<Integer> data = new ArrayList<Integer>();
		try {
			Class.forName("com.mysql.jdbc.Driver");
			Connection con = DriverManager.getConnection(
					"jdbc:mysql://localhost:3306/dengueprediction", "root", "root");

			StringBuilder sbSql = new StringBuilder("SELECT distinct(year) as year FROM PredictionData order by year");
			
			//sbSql.append(" GROUP BY c.lng,c.lat " );
			PreparedStatement oPrStmt = con
					.prepareStatement(sbSql.toString());
			ResultSet oRs =  oPrStmt.executeQuery();
			
			while(oRs.next()){
				data.add(oRs.getInt("year"));
			}
			oRs.close();
			oPrStmt.close();
			con.close();
		} catch (Exception e) {
			e.printStackTrace();
//			/System.out.println(e);
		}
		
		return data;
	}
	
	public HashMap<String,Integer> getWeekData(){
		HashMap<String,Integer> data = new HashMap<String,Integer>();
		try {
			Class.forName("com.mysql.jdbc.Driver");
			Connection con = DriverManager.getConnection(
					"jdbc:mysql://localhost:3306/dengueprediction", "root", "root");

			StringBuilder sbSql = new StringBuilder("SELECT YEAR,MAX(WEEK)as week FROM PredictionData GROUP BY YEAR");
			
			//sbSql.append(" GROUP BY c.lng,c.lat " );
			PreparedStatement oPrStmt = con
					.prepareStatement(sbSql.toString());
			ResultSet oRs =  oPrStmt.executeQuery();
			
			while(oRs.next()){
				data.put(oRs.getString("year"),oRs.getInt("week"));
			}
			oRs.close();
			oPrStmt.close();
			con.close();
		} catch (Exception e) {
			e.printStackTrace();
//			/System.out.println(e);
		}
		
		return data;
	}
}

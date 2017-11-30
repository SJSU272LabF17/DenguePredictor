<%@page import="java.util.Map"%>
<%@page import="java.util.Iterator"%>
<%@page import="java.util.HashMap"%>
<%@page import="java.util.ArrayList"%>
<%@page import="org.json.JSONArray"%>
<%@page import="data.GetMapData"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
<title>Dengue Prediction</title>
<style type="text/css">

/* Always set the map height explicitly to define the size of the div
 * element that contains the map. */
#map {
  height: 100%;
}
/* Optional: Makes the sample page fill the window. */
html, body {
  height: 100%;
  margin: 0;
  padding: 0;
}
</style>

<%
String strYear = request.getParameter("year");
String strWeek = request.getParameter("week");
int week = 0;

if(strWeek!= null && strWeek.trim().length()>0){
	week = Integer.parseInt(strWeek);
}


GetMapData data = new GetMapData();
ArrayList<Integer> years =  data.getYearData();
if(strYear== null || strYear.trim().length()==0){
	strYear =String.valueOf(years.get(years.size()-1));	
}

JSONArray strData =  data.getDataArray(strYear, week);
%>

</head>
<body>
<form action="" id="dropdowns">
<table>
<tr>
<td> Year</td> 
<td>
<select name="year" id="year" value="<%=strYear%>" onchange="submitForm(1)">
<%
int length = years.size();
String str =  "";
boolean bFound = false;
while(length>0){
	str = String.valueOf(years.get(--length));
	if(str.equalsIgnoreCase(strYear) || (!bFound && length==1)){
		System.out.println(str+"  "+strYear);
		str = " Checked='Checked'";
		
		bFound = true;
	}else{
		str = "";
	}
	%>
	<option value="<%=years.get(length)%>" <%=str%>><%=years.get(length)%></option>
	<%
}
%>

</select>


 </td> 
<td>
Week
</td> 
<td>
<%
HashMap<String,Integer> weeks = data.getWeekData();
int weekCount = weeks.get(strYear);
%>
<select name="week" id="week" value="<%=week%>" onchange="submitForm(2)">
<option value="0">--All--</option>
<%
int i = 1;
while(i<=weekCount){
	
	%>
	<option value="<%=i%>"><%=i%></option>
	<%
	i++;
}
%>

</select>
</td> 
</tr>
</table>
</form>
<div id="map"></div>
<!-- Replace the value of the key parameter with your own API key. -->
<script src="jquery-3.2.1.min.js"></script>
</body>
<script type="text/javascript">
function initMap() {	
	  var map = new google.maps.Map(document.getElementById('map'), {
	    zoom: 3,
	    center: {lat: -3.74912, lng: -73.25383}
	  });

	  // Create an array of alphabetical characters used to label the markers.
	  var labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

	  // Add some markers to the map.
	  // Note: The code uses the JavaScript Array.prototype.map() method to
	  // create an array of markers based on a given "locations" array.
	  // The map() method here has nothing to do with the Google Maps API.
	  var markers = locations.map(function(location, i) {
	    return new google.maps.Marker({
	      position: location,
	      label: labels[i % labels.length]
	    });
	  });

	  // Add a marker clusterer to manage the markers.
	  var markerCluster = new MarkerClusterer(map, markers,
	      {imagePath: 'https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/m'});
	}
	
var locations =<%=strData.toString().replaceAll("\"", "")%>; 

</script>
<script async defer
src="https://maps.googleapis.com/maps/api/js?key=AIzaSyD4uprGL8Qv17u53-b5y1wT331VGJhfi2k&callback=initMap">
</script>
<script src="https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/markerclusterer.js"></script>
<script>
$('#year').val('<%=strYear%>');
$('#week').val('<%=week%>');
var submitForm = function(source){
	if(source == 1){
		$('#week').val('0');
	}
	$('#dropdowns').submit();
}
</script>
</html>